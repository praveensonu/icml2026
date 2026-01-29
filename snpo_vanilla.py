# 1. export CUDA_VISIBLE_DEVICES=4,5
# 2. accelerate launch --multi_gpu --num_processes 2 simnpo.py

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from config import Config_2
from peft import  LoraConfig, get_peft_model
from data_module import SingleDataset
from collators import custom_data_collator
from utils import find_all_linear_names
from simnpo_utils import SNPO
from accelerate import Accelerator
import pandas as pd
from template import LLAMA3_CHAT_TEMPLATE



accelerator = Accelerator()

cfg = Config_2() #Config()




def read_file(path):
    if path.endswith('.csv'):
        df = pd.read_csv(path)
    elif path.endswith('.json'):
        df = pd.read_json(path)
    elif path.endswith('.parquet'):
        df = pd.read_parquet(path)
    return df

print('loading the forget, retain')
forget = read_file(cfg.forget_path)


print('forget shape:', forget.shape)

print(f"\nLoading the Tokenizer from {cfg.model_id}")
tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token = cfg.access_token)
tokenizer.pad_token = '<|finetune_right_pad_id|>'

print(f"\nLoading the Model {cfg.model_id}")
model = AutoModelForCausalLM.from_pretrained(cfg.model_id, 
                                             torch_dtype = torch.bfloat16, 
                                             token=cfg.access_token,
                                                device_map='auto'
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
# print('forget question and answer\n',forget['question'][0], forget['answer'][0])
# print('\n\nretain question and answer\n',retain['question'][0], retain['answer'][0])

cfg.save_dir = f'/raid/p.bushipaka/coreset/outputs/mix/{cfg.loss_type}_{cfg.ds_type}_{cfg.max_steps}'
print(f'Saving the model to {cfg.save_dir}')

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

print('\nlength of the dataset',len(dataset))


trainer = SNPO(
    model = model,
    args = training_args,
    train_dataset = dataset,
    tokenizer = tokenizer,
    data_collator = custom_data_collator,
)

trainer.train()

accelerator.wait_for_everyone()
model.save_pretrained(cfg.save_dir)
tokenizer.save_pretrained(cfg.save_dir)
print(f'\nForget LoRA adapter saved at {cfg.save_dir}')
