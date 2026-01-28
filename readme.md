# EFFECTIVE UNLEARNING IN LLMS RELIES ON THE RIGHT DATA RETENTION STRATEGY

### Abstract
LLM Unlearning methods rely on a retain set
to preserve utility while forgetting the undesired
information, yet how the choice of retain data
affects unlearning outcomes remains poorly understood. In this work, we present an empirical study of retain-set selection, focusing on how
data geometry interacts with different unlearning objectives. Across heterogeneous unlearning
setups and two unlearning algorithms, we show
that retain set properties such as representation
variance and size are strongly associated with
the forget-utility trade-off, but in algorithm dependent ways. We further introduce Represen-
tation Shift, a controlled retain-set construction
procedure, which we use as a diagnostic tool
to probe these associations. Through these controlled experiments, we find that gradient-based
unlearning is sensitive to representation variance,
whereas preference-based unlearning is primarily constrained by retain-set size and forget-set
diversity. Our results suggest that retain-set selection requires algorithm aware design rather than
one-size-fits-all heuristics.


### Create Environment
```bash
conda create -n corest python=3.11
conda activate coreset
pip install -r requirements.txt
```



### Finetune

To reproduce the results, the first step is to finetune the `Llama3.1-8B Instruct` model. We fine-tuned the model for 10 epochs with maximum learning rate of `2e-5` and batch size of 32. We used the original `meta-llama/Llama-3.1-8B-Instruct` HF repo. If you use the same hf repo, please update **access token** in the `Config_ft` class from ```config.py``` file. We need to finetune on two datasets giving us two different models (WPU, Mix). Based on this please select the right data path

```bash
Step 1: Change the max_length based on the dataset (WPU = 256, Mix = 512)
Step 2: Change the dataset path correctly.
python finetune.py
```

```bash
Step 1: Update the loss_type and exp_type (both pre_unlearning) in the Config/Config2 classes in config.py file
Step 2: Update line 23 in eval.py file based on the dataset (cfg = Config() for WPU, cfg = Config2() for Mix).
python eval.py
```


### Unlearn

There are overall 40 experiments just with Gradient difference. The datasets used for the experiments can be found in the `/data` folder. There are two Config classes in the ```config.py```. `Config` for **WPU** and `Config2` for **Mix** datasets. Based on your experiments, please select the right Config Class and. 


```bash
Step 1: Update the self.loss_type and exp_type in Config Class of config.py to gd
Step 2: Select the retain dataset type (mod_5, etc) and update the self.retain_path in the class
Step 3: Select the appropriate epochs from the paper (In the Appendix, Table:4) for the experiment.
Step 4: Please check if the model_id path is correct (finetuned version WPU/Mix).
Step 5: Change the config line code in the gd.py file (line 21, cfg = Config() for WPU, cfg = Config2() for Mix).
```


For gradient difference 

```bash
python gd.py
```
Without changing anything in the config, please follow the next steps for the evaluation. 

```bash
Step 1: Update line 23 in eval.py file based on the dataset (cfg = Config() for WPU, cfg = Config2() for Mix).
python eval.py
```


The above experiments can be reproduced on **multiple GPUs** too (except the evaluation). 

```bash
Step 1: Comment the os.environ['CUDA_VISIBLE_DEVICES'] = '1' in the files (finetune, gd). Lets say we do gd.py here
export CUDA_VISIBLE_DEVICES=0,1 
accelerate launch --multi_gpu --num_processes 2 gd.py
```

The code for **Coresets** selection is in the notebook, however reproducing that code requires extra steps (such as warmup runs for GRAND, semantic & syntactic calculation etc). Running the notebooks is not enough, it requires changes in the Config classes and doing the warmups for GRAND, calculating semantic and syntactic scores with sentence transformers model and edit distance etc. We already provide the selected coresets in the data folder for convenience. 



### Hyperparameter optimization 

For our experiments we actually used `gd_hpo.py` file for hyperparameter optimization, however this is extrememly time consuming. We suggest to use the epochs Table from the Paper and just run `gd.py` file. If you'd like to run the `gd_hpo.py`, please update the Config line and also update the **num_epochs** on line 69 in the file. Then you can use either multi-gpu or single gpu set up.

