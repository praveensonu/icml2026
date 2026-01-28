# 1. export CUDA_VISIBLE_DEVICES=4,5
# 2. accelerate launch --multi_gpu --num_processes 2 simnpo.py

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' #if you use multi gpu, comment ths out

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from config import Config_snpo
from peft import  LoraConfig, get_peft_model
from data_module import DualDataset, DualDatasetSeeded, DualDatasetSemantic
from collators import custom_gd_collator_forget
from utils import find_all_linear_names, read_file
from simnpo_utils import SimNPO
from accelerate import Accelerator
import pandas as pd
from template import LLAMA3_CHAT_TEMPLATE
from tabulate import tabulate


accelerator = Accelerator()

cfg = Config_snpo() 



metrics = [
     ("Data type", f'{cfg.data_type}'),
    ("Forgetting Experiment", f'{cfg.ds_type}'),
    ("Unlearning Loss", f'{cfg.loss_type}'),
    ("K selection", f'{cfg.k}'),
    ("Forget Path",   f'{cfg.forget_path}'),
    ("Retain Path",   f'{cfg.retain_path}'),
    ("SimNPO Delta",   f'{cfg.delta}'),
    ("SimNPO Beta",   f'{cfg.beta}'),
    ("Number of Steps",   f'{cfg.max_steps}'),
]


print("\n\n============ Check List ============\n")
print(tabulate(metrics, headers=["Metric", "Value"], tablefmt="github"))


# def read_file(path):
#     if path.endswith('.csv'):
#         df = pd.read_csv(path)
#     elif path.endswith('.json'):
#         df = pd.read_json(path)
#     elif path.endswith('.parquet'):
#         df = pd.read_parquet(path)
#     return df

print('loading the forget, retain')
forget = read_file(cfg.forget_path)
retain = read_file(cfg.retain_path)


# if cfg.k == 1.0:
#     assert forget.shape[0] == retain.shape[0], "For k=1, forget and retain sets must be of same size"
# if cfg.k == 2.0:
#     assert forget.shape[0] * 2 == retain.shape[0], "For k=2, retain set must be double the forget set size"
# if cfg.k == 5.0:
#     assert forget.shape[0] * 5 == retain.shape[0] , "For k=5, forget set must be double the retain set size"


print('forget shape:', forget.shape)
print('retain shape:', retain.shape)


print(f"\nLoading the Tokenizer from {cfg.model_id}")
tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token = cfg.access_token)
tokenizer.pad_token = '<|finetune_right_pad_id|>'

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
retain = make_template_format(retain)



# ------- Training Arguments ---------
training_args = TrainingArguments(
    output_dir = cfg.save_dir,
    overwrite_output_dir= True,
    learning_rate = cfg.lr,
    per_device_train_batch_size= cfg.batch_size, 
    #num_train_epochs= None,
    max_steps = cfg.max_steps,
    save_strategy= 'steps',
    save_steps= cfg.save_steps,
    weight_decay = cfg.weight_decay,
    logging_dir = f'{cfg.save_dir}/logs',
    eval_strategy= 'no',
    label_names = ['labels'],
    bf16 = True,
    gradient_accumulation_steps= cfg.gradient_accumulation_steps,
    ddp_find_unused_parameters=False,
)

if 'semantic' in cfg.data_type or 'syntactic' in cfg.data_type:
    dataset = DualDatasetSemantic(forget_data = forget,
                               retain_data = retain,
                               tokenizer =tokenizer,
                               max_length = cfg.max_length)
    print(f'\nUsing Dual Dataset Semantic for {cfg.data_type}')
else:
    dataset = DualDatasetSeeded(forget_data = forget,
                            retain_data = retain,
                            tokenizer = tokenizer,
                            max_length = cfg.max_length,
                            seed=42)
    print(f'\nUsing Dual Dataset Seeded for {cfg.data_type}')
print('\nlength of the dataset',len(dataset))

if cfg.loss_type == 'snpo':
    print('\nUsing Standard SimNPO Loss')
    trainer = SimNPO(
        model = model,
        args = training_args,
        train_dataset = dataset,
        tokenizer = tokenizer,
        data_collator = custom_gd_collator_forget,
    )

trainer.train()

accelerator.wait_for_everyone()
model.save_pretrained(cfg.save_dir)
tokenizer.save_pretrained(cfg.save_dir)
