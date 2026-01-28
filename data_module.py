from torch.utils.data import Dataset
import torch
import pandas as pd
import random

def convert_raw_data_to_model_qa(tokenizer, max_length,  question, answer):
    question = str(question)
    answer = str(answer)
    full_text = question + answer + tokenizer.eos_token
    num_question_tokens = len(tokenizer(question, add_special_tokens=False)['input_ids']) #this is important, we 
    encoded = tokenizer(
        full_text,
        add_special_tokens=False, #this is important, we keep false cause we already added the special tokens from template
        max_length=max_length,
        truncation=True,
    )
    input_ids = encoded['input_ids']
    pad_length = max_length - len(input_ids)
    pad_input_ids = encoded['input_ids']  + [tokenizer.pad_token_id] * pad_length
    pad_attention_mask = [1] * len(input_ids) + [0] * pad_length

    labels = list(input_ids) + [-100] * pad_length

    #change label to -100 for question tokens, including assistant header and end of header.
    for i in range(num_question_tokens): labels[i] = -100
    assert len(pad_input_ids) == max_length
    assert len(labels) == max_length
    assert len(pad_attention_mask) == max_length
    return torch.tensor(pad_input_ids),torch.tensor(labels),torch.tensor(pad_attention_mask)



# TOFU implementation
class DualDatasetRandom(Dataset):
    """
    TOFU way of implementation.

    Args:
        forget_data (pd.DataFrame): DataFrame for forgetting.
        retain_data (pd.DataFrame): DataFrame for retaining.
        tokenizer: tokenizer instance to process text.
        max_length (int): maximum sequence length.
        question_key (str): column name for questions.
        answer_key (str): column name for answers.
    """
    def __init__(self, forget_data, retain_data, tokenizer, max_length,
                 question_key = 'question',
                 answer_key = 'answer'):
        self.forget = forget_data.reset_index(drop=True)
        self.retain = retain_data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.forget)

    def __getitem__(self, idx):
        forget_idx = idx
        retain_idx = torch.randint(0, len(self.retain), (1,)).item()

        forget_data = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length,
            self.forget.iloc[forget_idx][self.qk],
            self.forget.iloc[forget_idx][self.ak],
        )

        retain_data = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length,
            self.retain.iloc[retain_idx][self.qk],
            self.retain.iloc[retain_idx][self.ak],
        )

        return (forget_data, retain_data)
    

    
class DualDatasetSeeded(Dataset):
    def __init__(
        self,
        forget_data,
        retain_data,
        tokenizer,
        max_length,
        seed: int,
        question_key='question',
        answer_key='answer',
    ):
        self.forget = forget_data.reset_index(drop=True)
        self.retain = retain_data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qk = question_key
        self.ak = answer_key
        
        random.seed(seed)  # Set seed for reproducibility

    def __len__(self):
        return len(self.retain)

    def __getitem__(self, idx):
        forget_idx = idx % len(self.forget)
        retain_idx = random.randint(0, len(self.retain) - 1)  # Random each time

        forget_data = convert_raw_data_to_model_qa(
            self.tokenizer,
            self.max_length,
            self.forget.iloc[forget_idx][self.qk],
            self.forget.iloc[forget_idx][self.ak],
        )

        retain_data = convert_raw_data_to_model_qa(
            self.tokenizer,
            self.max_length,
            self.retain.iloc[retain_idx][self.qk],
            self.retain.iloc[retain_idx][self.ak],
        )

        return forget_data, retain_data


class DualDatasetSemantic(Dataset):
    def __init__(self, forget_data, retain_data, tokenizer, max_length,
                 question_key='question',
                 answer_key='answer'):
        self.forget = forget_data.reset_index(drop=True)
        self.retain = retain_data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qk = question_key
        self.ak = answer_key
        
        # Map each forget id to a LIST of retain indices
        from collections import defaultdict
        self.id_to_retain_indices = defaultdict(list)
        for idx, row in self.retain.iterrows():
            self.id_to_retain_indices[row['id_f']].append(idx)

    def __len__(self):
        return len(self.forget)

    def __getitem__(self, idx):
        forget_idx = idx
        forget_id = self.forget.iloc[forget_idx]['id']
        
        # Randomly sample one retain sample from all associated ones
        retain_indices = self.id_to_retain_indices[forget_id]
        retain_idx = random.choice(retain_indices)

        forget_data = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length,
            self.forget.iloc[forget_idx][self.qk],
            self.forget.iloc[forget_idx][self.ak],
        )

        retain_data = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length,
            self.retain.iloc[retain_idx][self.qk],
            self.retain.iloc[retain_idx][self.ak],
        )

        return (forget_data, retain_data)



# for finetuning and gradient ascent
class SingleDataset(Dataset):
    def __init__(self, forget_data,
                 tokenizer,
                 max_length=512,
                 question_key = 'question',
                 answer_key = 'answer'):
        """
        Initializes the dataset for gradient ascent finetuning

        Args:
            data_path (str): path to the data file. csv file containing columns 'question' and 'answer'
            tokenizer (transformers.PreTrainedTokenizer): tokenizer to process the input
            max_length (int, optional): maximum sequence length for tokenization. Defaults to 512.
            template_format (str, optional): format template for structuring input
        """
        self.data = forget_data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data.iloc[idx][self.qk]
        answer = self.data.iloc[idx][self.ak]
        return convert_raw_data_to_model_qa(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            question=question,
            answer=answer
        )



#Cyclic implementation, this is for the discarded experiments in the paper (page 30)
class DualDataset(Dataset):
    """
    Dataset class for creating data for forget and retain (used by gradient difference)

    Args:
        forget_data (pd.DataFrame): DataFrame containing 'question' and 'answer' columns for forgetting
        retain_data (pd.DataFrame): DataFrame containing 'question' and 'answer' columns for retaining
        tokenizer: tokenizer instance to process text
        max_length (int): maximum sequence length
        template_format (str, optional): format template for structuring input

    Returns:
        Tuple of forget and retain samples:
        (
            (forget_input_ids, forget_labels, forget_attention_mask),
            (retain_input_ids, retain_labels, retain_attention_mask)
        )
    """
    def __init__(self, forget_data, retain_data, tokenizer, max_length,
                 question_key = 'question',
                 answer_key = 'answer'):
        self.forget = forget_data.reset_index(drop=True)
        self.retain = retain_data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qk = question_key
        self.ak = answer_key
    def __len__(self):
        return max(len(self.forget), len(self.retain))

    def __getitem__(self, idx):
        # Cyclic rotation of data
        forget_idx = idx % len(self.forget)
        retain_idx = idx % len(self.retain)

        forget_data = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length,
            self.forget.iloc[forget_idx][self.qk],
            self.forget.iloc[forget_idx][self.ak],
        )

        retain_data = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length,
            self.retain.iloc[retain_idx][self.qk],
            self.retain.iloc[retain_idx][self.ak],
        )

        return (forget_data, retain_data)
