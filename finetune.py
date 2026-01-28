#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '5'

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import torch
import pandas as pd
from config import Config_ft
from datasets import load_dataset
from peft import  LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from accelerate import  Accelerator
from utils import find_all_linear_names, read_file
from template import LLAMA3_CHAT_TEMPLATE
from data_module import SingleDataset
from collators import custom_data_collator





cfg = Config_ft()
data = pd.read_file(cfg.data_path)
print(data.shape)


data['question'] = data['question'].apply(lambda x : LLAMA3_CHAT_TEMPLATE.format(question = x))
print('\n\n',data['question'][0])

accelerator = Accelerator()

tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token = cfg.access_token)
tokenizer.pad_token = '<|finetune_right_pad_id|>'

model = AutoModelForCausalLM.from_pretrained(
    cfg.model_id, 
    device_map = 'cuda',
    torch_dtype = torch.bfloat16, 
    token=cfg.access_token,
    low_cpu_mem_usage=True,
    # attn_implementation ='flash_attention_2',
)


Lora_config = LoraConfig(
    r = cfg.LoRA_r,
    lora_alpha = cfg.LoRA_alpha,
    lora_dropout= cfg.LoRA_dropout,
    target_modules = find_all_linear_names(model),
    bias = 'none',
    task_type = 'CAUSAL_LM',
)

model.config.use_cache = False

model = get_peft_model(model, Lora_config)

model.print_trainable_parameters()

dataset =SingleDataset(data, tokenizer, max_length = cfg.max_length)

args = TrainingArguments(
    per_device_train_batch_size = cfg.batch_size,
    learning_rate = cfg.lr,
    bf16 = True,
    num_train_epochs = cfg.num_epochs,
    weight_decay = cfg.weight_decay,
    logging_dir = f'{cfg.save_dir}/logs',
    eval_strategy= 'no',
    gradient_accumulation_steps = cfg.gradient_accumulation_steps,
    save_strategy = 'epoch',
    save_total_limit = 1,
    output_dir = cfg.save_dir,
    gradient_checkpointing=False,
    ddp_find_unused_parameters=False,

)

trainer = Trainer(
    model = model,
    args = args,
    train_dataset = dataset,
    tokenizer = tokenizer,
    data_collator = custom_data_collator,
)

trainer.train()


model = model.cpu() 
model = model.merge_and_unload()

print(f"Model and tokenizer saved to {cfg.save_dir}")
model.save_pretrained(cfg.save_dir)
tokenizer.save_pretrained(cfg.save_dir)



