from controller import TASK_Trainer
import os
import joblib
import random
random.seed(1509)

def run_train():
    # embbeded features of each images in each task id
    # dict[task_id]['images'] = list of images within the task_id
    # dict[task_id]['features'] = nympy array (n_images, ft_dim=1024) features of each images
    
    a = joblib.load('../joblib_files/task_features_train_RUN_0_Unfreeze.joblib')
    task_id_list = list(a.keys())
    random.shuffle(task_id_list)
    train_portion = 0.7
    numb_train = int(train_portion * len(task_id_list))
    train_task = task_id_list[0:numb_train]
    val_task = task_id_list[numb_train:]            
    train_dict = dict()
    val_dict = dict()
    for key, val in a.items():
        if key in train_task:
            train_dict[key] = val
        else:
            val_dict[key] = val
    
    BATCH_SIZE = 64
    MAX_EPOCH = 100
    FT_DIM = 1024
    TRAIN_DICT = train_dict
    VAL_DICT = val_dict
    HIDDEN_SIZE = [512, 256]
    MODEL_NAME = 'TASK_512_256'
    DATA_MODE = 'mean'
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
    model_info['ft_dim'] = FT_DIM
    model_info['train_dict'] = TRAIN_DICT
    model_info['val_dict'] = VAL_DICT
    model_info['hidden_size'] = HIDDEN_SIZE
    model_info['data_mode'] = DATA_MODE
    model_info['save_dir'] = SAVE_DIR
    model_info['optimizer'] = OPTIM
    model_info['model_name'] = MODEL_NAME
    model_info['dropout'] = DROPOUT
    model_info['learning_rate'] = LR
    model_info['batch_norm'] = BATCH_NORM
    
    if CHECKPOINT is not None:
        model_info['checkpoint'] = CHECKPOINT
        
    task_trainer = TASK_Trainer(json_info=model_info)
    
    print(f"===== START TRAINING ON MART DATA =====")
    task_trainer.train()
    
    
run_train()