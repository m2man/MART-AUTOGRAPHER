from controller import Trainer
import os
import joblib
import pandas as pd
import random
random.seed(1509)

def run_train():
    # embbeded features of each images in each task id
    # dict[task_id]['images'] = list of images within the task_id
    # dict[task_id]['features'] = nympy array (n_images, ft_dim=1024) features of each images
    
    a = joblib.load('tabular_only_csv.joblib')
    a = a.sample(frac=1)
    train_portion = 0.8
    train_numb = int(train_portion * len(a))
    train_df = a.iloc[0:train_numb, :]
    val_df = a.iloc[train_numb:, :]
    
    BATCH_SIZE = 64
    MAX_EPOCH = 100
    INPUT_DIM = len(train_df.columns) - 1 # last column is the label
    TRAIN_DF = train_df
    VAL_DF = val_df
    LAYER_DIM = [1000, 500, 128]
    MODEL_NAME = 'CSV_TABULAR'
    CHECKPOINT = None 
    SAVE_DIR = 'RUN_0'
    OPTIM = 'Adam'
    DROPOUT = 0.5
    LR = 0.005
    BATCH_NORM = True
    
    if not os.path.exists(f'{SAVE_DIR}'):
        print(f'Creating {SAVE_DIR} folder')
        os.makedirs(f'{SAVE_DIR}')
    
    model_info = {}
    model_info['batch_size'] = BATCH_SIZE
    model_info['max_epoch'] = MAX_EPOCH
    model_info['input_dim'] = INPUT_DIM
    model_info['train_df'] = TRAIN_DF
    model_info['val_df'] = VAL_DF
    model_info['layer_dim'] = LAYER_DIM
    model_info['save_dir'] = SAVE_DIR
    model_info['optimizer'] = OPTIM
    model_info['model_name'] = MODEL_NAME
    model_info['dropout'] = DROPOUT
    model_info['learning_rate'] = LR
    model_info['batch_norm'] = BATCH_NORM
    
    if CHECKPOINT is not None:
        model_info['checkpoint'] = CHECKPOINT
        
    trainer = Trainer(json_info=model_info)
    
    print(f"===== START TRAINING ON MART DATA =====")
    trainer.train()
    
    
run_train()