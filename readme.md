# Effective LLM Unlearning with right retention data

### Abstract
LLM Unlearning methods rely on a retain set to preserve utility while forgetting the undesired information, yet how the choice of retain data affects unlearning outcomes remains poorly understood. In this work, we present an empirical study of retain-set selection, focusing on how data geometry interacts with different unlearning objectives. Across heterogeneous unlearning setups and two unlearning algorithms, we show that retain set properties such as representation variance and size are strongly associated with the forget-utility trade-off, but in algorithm dependent ways. We further introduce Representation Shift, a controlled retain-set construction procedure, which we use as a diagnostic tool
to probe these associations. Through these controlled experiments, we find that gradient-based unlearning is sensitive to representation variance, whereas preference-based unlearning is primarily constrained by retain-set size and forget-set diversity. Our results suggest that retain-set selection requires algorithm aware design rather than one-size-fits-all heuristics.




### Create Environment

We use CUDA 12.1 for our experiments, please install the respective cuda toolkit 

```bash
conda create -n corest python=3.11
conda activate coreset
pip install -r requirements.txt
```



### Finetune

To reproduce the results, the first step is to finetune the `Llama3.1-8B Base` model. We fine-tuned the model for 10 epochs with maximum learning rate of `2e-5` and batch size of 32. We used the original `meta-llama/Llama-3.1-8B` HF repo. If you use the same hf repo, please update **access token** in the `Config_ft` class from ```config.py``` file. We need to finetune the model on the full dataset. Based on this please select the right data path

In the config file, we have Config_ft -> finetuning, Config_gd -> to run graddiff experiments, Config_snpo -> SimNPO, Config_eval -> to run evals. Mostly, you need to change the ds_type, data_type, loss_type in these configs. 
ds_type refers to data setup. we have 3 ds_types -> {ds_1, ds_2, ds_3}
loss_type refers to algorithm. gd and snpo
data_type refers to data selection methods {el2n, moderate, random, semantic, syntactic}. However, you need to add _1,_2,_5 (Ex: el2n_1 refers to el2n with $k$ size of 1, same goes for 2,5). 
Additional data_type are {gd_ortho_{k}, snpo_ortho_{k}...}

So, any dataselection path goes to `./data/ds_type (ds_1 or ds_2 or ds_3)/data_type`.


```bash
Step 1: Add the HF key in the Config_ft.
python finetune.py
```

```bash
Step 1: Update the loss_type and ds_type (both pre_unlearning) in the Config_eval classes in config.py file
python eval.py
```


### Unlearn GradDiff

There are overall 45 (excluding ortho and hard) experiments just with Gradient difference. The datasets used for the experiments can be found in the `/data` folder. Use ```Config_gd``` from ```config.py```. Update the `ds_type` and `data_type`.


```bash
python gd.py
```
Please change the loss_type (gd), ds_type and data_type in the Config_eval in the ```config.py```

```bash
python eval.py
```

### Unlearn SimNPO

There are overall 45 (excluding ortho and hard) experiments just with SimNPO. The datasets used for the experiments can be found in the `/data` folder. Use ```Config_snpo``` from ```config.py```. Update the `ds_type` and `data_type`.


```bash
python snpo.py
```
Please change the loss_type (snpo), ds_type and data_type in the Config_eval in the ```config.py```

```bash
python eval.py
```

The above experiments can be reproduced on **multiple GPUs** too (except the evaluation). 

```bash
Step 1: Comment the os.environ['CUDA_VISIBLE_DEVICES'] = '1' in the files (finetune, gd). Lets say we do gd.py here
export CUDA_VISIBLE_DEVICES=0,1 
accelerate launch --multi_gpu --num_processes 2 gd.py
```

The code for **Coresets** selection is in the notebook, however reproducing that code requires extra steps (such as, semantic & syntactic calculation etc). Running the notebooks is not enough, it requires changes in the Config classes, calculating semantic and syntactic scores with sentence transformers model and edit distance etc. We already provide the selected coresets in the data folder for convenience. 


You may want to do vanilla unlearning for Representation shift (ascent and snpo). please use grad_ascent.py and snpo_vanilla.py for this. Update config_2 in config.py




