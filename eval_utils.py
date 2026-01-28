import torch
import numpy as np
from transformers import PreTrainedTokenizer
import torch.nn.functional as F
from scipy.stats import hmean
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from tqdm.auto import tqdm
import warnings
from accelerate import Accelerator
from typing import List, Tuple
from collections import Counter
import math

warnings.filterwarnings("ignore")

accelerator = Accelerator()



def eval_rouge_recall(gen_outputs, ground_truths):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(gen_outputs, ground_truths)

    return rouge_scores['rouge1'].recall, rouge_scores['rougeL'].recall



def eval_cosine_similarity_batched(gen_outputs: List[str], ground_truths: List[str], model: SentenceTransformer, batch_size: int = 32):
    """
    Calculates cosine similarity for all pairs in a batched and efficient manner.
    """
    with torch.no_grad():
        gen_embeddings = model.encode(
            gen_outputs, 
            batch_size=batch_size, 
            show_progress_bar=True, 
            convert_to_tensor=True
        )
        gt_embeddings = model.encode(
            ground_truths, 
            batch_size=batch_size, 
            show_progress_bar=True, 
            convert_to_tensor=True
        )
        pairwise_scores = torch.diag(util.cos_sim(gen_embeddings, gt_embeddings))
        return torch.clamp(pairwise_scores, min=0).tolist()



@torch.no_grad()
def get_probs_ppl(
    question: str,
    answer: str,
    model,
    tokenizer: PreTrainedTokenizer,
    device,
):
    full_text = question + answer + tokenizer.eos_token
    questions_encoded = tokenizer(question, add_special_tokens=False, return_tensors='pt').to(device)
    num_questions = questions_encoded['input_ids'].size(1)
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        return_tensors='pt',
    ).to(device)

    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    labels = input_ids.clone()
    labels[0,:num_questions] = -100
    out = model(input_ids, attention_mask= attention_mask,labels = labels)
    loss = out.loss

    #T = (labels[0] != -100).sum().item()
    gmp = float(torch.exp(-loss)) #goemetric mean of probs,
    ppl = float(torch.exp(loss)) #perplexity  
    return gmp, ppl   



@torch.no_grad()
def eval_truth_ratio(question : str, true_answer : str, false_answers: List[str],model, 
                     tokenizer, device):
    """
    Calculate truth ratio: mean(P(false_answers))/P(correct_answer)
    """
    true_prob,_ = get_probs_ppl(question, true_answer, model, tokenizer, device)
    false_probs = []
    for f in false_answers:
        probs,_ = get_probs_ppl(question, f, model, tokenizer, device)
        false_probs.append(probs)
    mean_false_prob = np.mean(false_probs) if false_probs else 0.0
    truth_ratio_val = mean_false_prob/ (true_prob + 1e-10)
    tr_score = np.minimum(truth_ratio_val, 1 / truth_ratio_val)
    return tr_score
    

@torch.no_grad()
def generate_outputs(question :str, model, tokenizer, device, max_new_tokens: int = 50):
    inputs = tokenizer(
        question, 
        return_tensors="pt", 
        add_special_tokens=False
    ).to(device)
    out = model.generate(
        **inputs,
        max_new_tokens = max_new_tokens,
        do_sample = False,
        return_dict_in_generate = False
    )
    full_seq = out[0]
    input_ids = inputs['input_ids']
    gen_ids = full_seq[input_ids.size(1):]
    answer = tokenizer.decode(
        gen_ids, 
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return answer



def compute_fq_scores(df, model, tokenizer, embedding_model, device):
    
    gen_answers, probas, rougels, truth, ppls = ([] for _ in range(5))
    ground_truths_for_batch = [] 

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing forget efficacy", disable=not accelerator.is_main_process):
        q, a, f_a_list, num_tokens,para_q, para_ans = row['question'], row['answer'], row['perturbed_answers'], row['num_tokens'], row['paraphrased_question'], row['paraphrased_answer']
        

        probs, ppl = get_probs_ppl(q, a, model, tokenizer, device=device)
        gen = generate_outputs(q, model, tokenizer, device=device, max_new_tokens=num_tokens)
        _, rl = eval_rouge_recall(gen, a) 
        tr = eval_truth_ratio(para_q, para_ans, f_a_list, model, tokenizer, device)
        
    
        gen_answers.append(gen)
        ground_truths_for_batch.append(a) 
        
        probas.append(probs)
        rougels.append(rl)
        ppls.append(ppl)
        truth.append(tr)
        
    print("Calculating cosine similarity in a batch...")
    cos_sims = eval_cosine_similarity_batched(gen_answers, ground_truths_for_batch, embedding_model)

    df['gen_answer'] = gen_answers
    df['probs']      = probas
    df['rouge_l']    = rougels
    df['truth']      = truth
    df['ppl']        = ppls
    df['cos_sims']   = cos_sims 
    
    # Now calculate the averages
    avg_cos_sims = np.mean(cos_sims)
    all_scores = np.array([1.0 - np.mean(probas), 1.0 - np.mean(rougels), 1.0 - np.mean(truth)])
    forget_quality = hmean(all_scores)
    #forget_efficacy = 1.0 - np.mean(all_scores)
    all_scores = np.append(all_scores, avg_cos_sims)
    avg_ppl = np.mean(ppls)
    
    print(f'forget_quality: {forget_quality:.4f}')
    return df, all_scores, forget_quality, avg_ppl


def compute_mu_scores(df, model, tokenizer, embedding_model, device):
    gen_answers, probas, rougels, ppls, js_scores, syntactic = ([] for _ in range(6))
    ground_truths_for_batch = [] 

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing model utility", disable=not accelerator.is_main_process):
        q, a, num_tokens = row['question'], row['answer'], row['num_tokens']
        
        probs, ppl = get_probs_ppl(q, a, model, tokenizer, device=device)
        gen = generate_outputs(q, model, tokenizer, device=device, max_new_tokens=num_tokens)
        _, rl = eval_rouge_recall(gen, a)
        # js_score = get_js_scores(a, gen)
        # syn_sim   = syntactic_similarity(a, gen)

        
        gen_answers.append(gen)
        ground_truths_for_batch.append(a) 
        probas.append(probs)
        rougels.append(rl)
        ppls.append(ppl)
        # js_scores.append(js_score)
        # syntactic.append(syn_sim)
        
    print("Calculating cosine similarity in a batch...")
    cos_sims = eval_cosine_similarity_batched(gen_answers, ground_truths_for_batch, embedding_model)

    df['gen_answer'] = gen_answers
    df['probs']      = probas
    df['rouge_l']    = rougels
    df['ppl']        = ppls
    df['cos_sims']   = cos_sims
    # df['js_scores']  = js_scores
    # df['syntactic']  = syntactic

    # all_scores = np.array([np.mean(probas), np.mean(rougels), 
    #                        np.mean(cos_sims), 1 - np.mean(js_scores), np.mean(syntactic)])
    all_scores = np.array([np.mean(probas), np.mean(rougels), np.mean(cos_sims)])
    mu = hmean(all_scores)
    avg_ppl = np.mean(ppls)
    
    print(f'mu: {mu:.4f}')
    return df, all_scores, mu, avg_ppl
    


  



