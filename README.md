# AuD (Are You Depressed ?)
AuD is model that analyze sentence and return depression intensity, related emotions, attention tokens 

We use language models to learn BWS and DSM-5 data built by refining data collected from Reddit Archive and AI-Hub. For user input speech, the BWS model performs depression intensity and the DSM-5 model performs detailed depression emotion classification.

_Model Architecture_   
![model architecture](https://user-images.githubusercontent.com/48609095/223356781-7e6dd680-9f92-4583-96bd-de4865ff857d.PNG)

***
### 1. How to use 
#### 1) Data curate 
we process reddit data in this repository: https://github.com/Jaedong95/Reddit

using dataset2.csv, we construct bws, dsm-5 data respectively 


#### 2) Train model 
We train BWS & DSM-5 model 
```bash 
$ python bws-train.py --data_path {$DATA_PATH} --base_model {$BASE_MODEL_PATH} --model_path {$MODEL_PATH} --config_path {$CONFIG_PATH} --log_path {$LOG_PATH} --config_file {$CONFIG_FILE}

$ python dsm-train.py --data_path {$DATA_PATH} --base_model {$BASE_MODEL_PATH} --model_path {$MODEL_PATH} --config_path {$CONFIG_PATH} --log_path {$LOG_PATH} --config_file {$CONFIG_FILE}
```

#### 3) Service model 
