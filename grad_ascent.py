import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from config import Config2
from peft import LoraConfig, get_peft_model 
from data_module import SingleDataset
from collators import custom_data_collator
from utils import find_all_linear_names
from forget_trainer import GATrainer
from accelerate import Accelerator
import pandas as pd
from template import LLAMA3_CHAT_TEMPLATE



accelerator = Accelerator()

cfg = Config2()


# ------- loading the datafiles-------------
try:
    cfg.loss_type=='ascent'
    print('Gradient Ascent Experiment')
except:
    print('Other Experiment')

def read_file(path):
    if path.endswith('.csv'):
        df = pd.read_csv(path)
    elif path.endswith('.json'):
        df = pd.read_json(path)
    elif path.endswith('.parquet'):
        df = pd.read_parquet(path)
    return df

forget = read_file(cfg.forget_path)

print('forget shape:', forget.shape)
print(f'The current experiment type is : {cfg.ds_type}')


# ------- Load the tokenizer ----------------
print(f"\nLoading the Tokenizer {cfg.model_id}")
tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token = cfg.access_token)
tokenizer.pad_token = '<|finetune_right_pad_id|>'


# ------- Load the model ----------------
print(f"\nLoading the Model {cfg.model_id}")
model = AutoModelForCausalLM.from_pretrained(cfg.model_id, 
                                             torch_dtype = torch.bfloat16, 
                                             token=cfg.access_token,
                                             #attn_implementation ='flash_attention_2'
                                             )
config = LoraConfig(
        r = cfg.LoRA_r,
        lora_alpha = cfg.LoRA_alpha,
        lora_dropout= cfg.LoRA_dropout,
        target_modules = find_all_linear_names(model),
        bias = 'none',
        task_type = 'CAUSAL_LM',
    )

print(f"{config.target_modules}")

# ------- wrapping the model with the LoRA configuration
model = get_peft_model(model, config)
model.print_trainable_parameters()
model.config.use_cache = False


# ------- creating template format for tokenization --------
def make_template_format(df):
    df['question'] = df['question'].apply(lambda x : LLAMA3_CHAT_TEMPLATE.format(question = x))
    return df

forget = make_template_format(forget)

print('forget question and answer\n',forget['question'][0],forget['answer'][0])




#------- Training Arguments ---------
training_args = TrainingArguments(
    output_dir = cfg.save_dir,
    overwrite_output_dir= True,
    learning_rate = cfg.lr,
    per_device_train_batch_size= cfg.batch_size, 
    num_train_epochs= cfg.num_epochs,
    weight_decay = cfg.weight_decay,
    logging_dir = f'{cfg.save_dir}/logs',
    eval_strategy= 'no',
    label_names = ['labels'],
    bf16 = True,
    gradient_accumulation_steps= cfg.gradient_accumulation_steps,
    ddp_find_unused_parameters=False,
)



dataset = SingleDataset(forget,
                        tokenizer,
                        max_length = cfg.max_length,
                        question_key='question',
                        answer_key='answer')

trainer = GATrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator = custom_data_collator,
    )

trainer.train()

accelerator.wait_for_everyone()
model.save_pretrained(cfg.save_dir)
tokenizer.save_pretrained(cfg.save_dir)
print(f'\nForget LoRA adapter saved at {cfg.save_dir}')
