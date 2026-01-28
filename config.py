# for wpu
class Config:
    def __init__(self):
        super(Config, self).__init__()
        self.loss_type      = 'gd' # change this with the experiment types provided above
        self.access_token   = '' # huggingface token
        self.model_id       = './outputs/wpu_llama' 
        self.LoRA_r         = 8
        self.LoRA_alpha     = 16
        self.LoRA_dropout   = 0.05
        self.lr             = 1e-5
        self.LoRa_targets   = ['v_proj', 'k_proj', 'up_proj', 'o_proj', 'gate_proj', 'q_proj', 'down_proj']
        self.batch_size     = 8 #please adjust this along with gradient_accumulation_steps for batch size
        self.gradient_accumulation_steps = 1 
        self.num_epochs     = 25
        self.overwrite_dir  = True
        self.weight_decay   = 0.01
        self.max_length     = 256
        self.exp_type       = 'semantic_5' #change this with the experiment types provided in the paper
        self.save_dir       = f'outputs/{self.loss_type}_{self.exp_type}_model' 
        self.retriever_model= 'thenlper/gte-small'
        self.forget_path    = './data/wpu_data/forget_1.csv' 
        self.retain_path    = './data/wpu_data/semantic/semantic_5.csv' #change this path
        self.retain_full    = './data/wpu_data/retain_100.csv'
        self.npo_beta       = 0.1
        self.npo_retain_alpha = 1.0
        self.npo_forget_gamma = 1.0
        

class Config_snpo:
    def __init__(self):
        super(Config_snpo, self).__init__()
        self.loss_type      = 'snpo' # change this with the experiment types provided above
        self.access_token   = '' 
        self.model_id       = '/outputs/llama_ft' # just check this path
        self.LoRA_r         = 8
        self.LoRA_alpha     = 16
        self.LoRA_dropout   = 0.05
        self.lr             = 2e-5
        self.LoRa_targets   = ['v_proj', 'k_proj', 'up_proj', 'o_proj', 'gate_proj', 'q_proj', 'down_proj']
        self.batch_size     = 1 # for 2 gpus, change this accordingly to you
        self.gradient_accumulation_steps = 4 #always batch size of 8
        self.num_epochs     = 1
        self.overwrite_dir  = True
        self.weight_decay   = 0.01
        self.max_length     = 512
        self.data_type      = 'moderate_1' #this is the data selection method used
        self.ds_type        = 'ds_1' #change this based on the dataset you are using ds_1, ds_2 etc
        self.save_dir       = f'/outputs/{self.loss_type}_{self.ds_type}_{self.data_type}' 
        self.retriever_model= 'thenlper/gte-small'
        self.max_steps      = 150
        self.save_steps     = 20
        self.gradient_res_path = '/raid/p.bushipaka/coreset/grad_stats'
        self.delta          = 0.0
        self.beta           = 3.5
        self.k              = 1.0
        
        # Dynamic paths based on ds_type and data_type
        self.base_data_dir  = './data'
        
    @property
    def forget_path(self):
        """Dynamically generate forget path based on ds_type"""
        # Extract the number from ds_type (e.g., 'ds_1' -> '1')
        ds_num = self.ds_type.split('_')[1]
        return f'{self.base_data_dir}/datasets/forget_{ds_num}.parquet'
    
    @property
    def retain_path(self):
        """Dynamically generate retain path based on ds_type and data_type"""
        return f'{self.base_data_dir}/{self.ds_type}/{self.data_type}.parquet'





## this is for finetuning
class Config_ft:
    def __init__(self):
        super(Config_ft, self).__init__()
        self.model_id       = 'meta-llama/Llama-3.1-8B'
        self.access_token   = '' # please add ur huggingface token 
        self.LoRA_r         = 64 
        self.LoRA_alpha     = 64 
        self.LoRA_dropout   = 0.05
        self.lr             = 2e-05 
        self.LoRa_targets   = ['v_proj', 'k_proj', 'up_proj', 'o_proj', 'gate_proj', ' q_proj', 'down_proj']
        self.batch_size     = 32 #please adjust this along with gradient_accumulation_steps for batch size
        self.gradient_accumulation_steps = 1 
        self.num_epochs     = 10
        self.overwrite_dir  = True
        self.weight_decay   = 0.01
        self.exp_type       = 'ckpt_desc'
        self.model_name    = 'llama_8b'
        self.save_dir       = '/outputs/llama_ft' 
        self.max_length     = 512 
        self.data_path      = './data/full_data.parquet' 
