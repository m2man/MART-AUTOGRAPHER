from controller import Trainer, Evaluator
import os
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
random.seed(1509)

def run_train():
    # embbeded features of each images in each task id
    # dict[task_id]['images'] = list of images within the task_id
    # dict[task_id]['features'] = nympy array (n_images, ft_dim=1024) features of each images
    
    a = joblib.load('tabular.joblib')
    a = a.drop(columns=['event_id', 'source', 'sub_id'])
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
    SAVE_DIR = 'INCLUDE_NA'
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
    result = trainer.train()
    
    return result
    
def run_kfold(k_fold=10):
    loss_val = np.zeros(k_fold)
    loss_train = np.zeros(k_fold)
    acc_val = np.zeros(k_fold)

    for k in range(k_fold):
        result = run_train()
        loss_val[k] = result['loss_val_save']
        loss_train[k] = result['loss_train_save']
        acc_val[k] = result['acc_val_save']

    print(loss_val)
    print(loss_train)
    print(acc_val)

    print(f"[LossTrain]Min-Max-Mean: {np.min(loss_train)}-{np.max(loss_train)}-{np.mean(loss_train)}")
    print(f"[LossVal]Min-Max-Mean: {np.min(loss_val)}-{np.max(loss_val)}-{np.mean(loss_val)}")
    print(f"[AccMean]Min-Max-Mean: {np.min(acc_val)}-{np.max(acc_val)}-{np.mean(acc_val)}")

def run_extract_features():
    a = joblib.load('tabular.joblib')
    event_id = list(a['event_id'])
    source = list(a['source'])
    sub_id = list(a['sub_id'])
    a = a.drop(columns=['event_id', 'source', 'sub_id'])
    if 'label' in list(a.columns):
        a = a.drop(columns=['label'])
    
    INPUT_DIM = len(a.columns)    
    LAYER_DIM = [1000, 500, 128]
    MODEL_NAME = 'CSV_TABULAR'
    CHECKPOINT = 'INCLUDE_NA/CSV_TABULAR-26082020-155927.pth.tar' 
    DROPOUT = 0.5
    BATCH_NORM = True
    
    model_info = {}
    model_info['input_dim'] = INPUT_DIM
    model_info['layer_dim'] = LAYER_DIM
    model_info['model_name'] = MODEL_NAME
    model_info['dropout'] = DROPOUT
    model_info['batch_norm'] = BATCH_NORM
    evaluator = Evaluator(json_info=model_info)
    
    print('Embedding features ...')
    embed_result = dict()
    for row in tqdm(range(len(a))):
        task_id = f"{sub_id[row]}_{source[row]}_{event_id[row]}"
        row_ft = a.iloc[row]
        row_ft = np.asarray(row_ft)
        embed = evaluator.extract_features(row_ft)
        embed = embed.cpu().numpy()
        embed_result[task_id] = embed
    
    joblib.dump(embed_result, 'tabular_embeded_ft.joblib')
    
run_extract_features()