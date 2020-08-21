from controller import Trainer
import os

def run_train():
    BATCH_SIZE = 64
    CROP_SIZE = 224 # DO NOT CHANGE THIS
    MAX_EPOCH = 200
    DATA_DIR = '/mnt/sda/hong01-data/MART_DATA/OUTPUT_MERGED/AUTOGRAPHER'
    TRAIN_TXT = '../dataset/all.txt'
    VAL_TXT = '../dataset/val_test.txt'
    CHECKPOINT = None 
    CHECKPOINT = 'RUN_0/KFOLD-EFFICIENT-B4-20082020-171342.pth.tar'
    SAVE_DIR = 'RUN_1'
    OPTIM = 'Adam'
    MODEL_NAME = 'KFOLD-EFFICIENT-B4'
    DROPOUT = 0.5
    LR = 0.001
    FREEZE = False
    HIDDEN_SIZE = 512
    BATCH_NORM = True
    
    if not os.path.exists(f'{SAVE_DIR}'):
        print(f'Creating {SAVE_DIR} folder')
        os.makedirs(f'{SAVE_DIR}')
    
    model_info = {}
    model_info['batch_size'] = BATCH_SIZE
    model_info['crop_size'] = CROP_SIZE
    model_info['max_epoch'] = MAX_EPOCH
    model_info['data_dir'] = DATA_DIR
    model_info['train_txt'] = TRAIN_TXT
    model_info['val_txt'] = VAL_TXT
    model_info['save_dir'] = SAVE_DIR
    model_info['optimizer'] = OPTIM
    model_info['model_name'] = MODEL_NAME
    model_info['dropout'] = DROPOUT
    model_info['learning_rate'] = LR
    model_info['freeze'] = FREEZE
    model_info['hidden_size'] = HIDDEN_SIZE
    model_info['batch_norm'] = BATCH_NORM
    
    if CHECKPOINT is not None:
        model_info['checkpoint'] = CHECKPOINT
        
    trainer = Trainer(json_info=model_info)
    
    print(f"===== START TRAINING ON MART DATA =====")
    trainer.train()
    
def run_kfold_validate():
    BATCH_SIZE = 64
    CROP_SIZE = 224 # DO NOT CHANGE THIS
    MAX_EPOCH = 200
    DATA_DIR = '/mnt/sda/hong01-data/MART_DATA/OUTPUT_MERGED/AUTOGRAPHER'
    TRAIN_TXT = '../dataset/all.txt'
    VAL_TXT = '../dataset/val_test.txt'
    #CHECKPOINT = None 
    CHECKPOINT = 'RUN_0/KFOLD-EFFICIENT-B4-20082020-171342.pth.tar'
    SAVE_DIR = 'RUN_KFOLD_VALIDATE'
    OPTIM = 'Adam'
    MODEL_NAME = 'KFOLD-EFFICIENT-B4'
    DROPOUT = 0.5
    LR = 0.001
    FREEZE = False
    HIDDEN_SIZE = 512
    BATCH_NORM = True
    
    if not os.path.exists(f'{SAVE_DIR}'):
        print(f'Creating {SAVE_DIR} folder')
        os.makedirs(f'{SAVE_DIR}')
    
    model_info = {}
    model_info['batch_size'] = BATCH_SIZE
    model_info['crop_size'] = CROP_SIZE
    model_info['max_epoch'] = MAX_EPOCH
    model_info['data_dir'] = DATA_DIR
    model_info['train_txt'] = TRAIN_TXT
    model_info['val_txt'] = VAL_TXT
    model_info['save_dir'] = SAVE_DIR
    model_info['optimizer'] = OPTIM
    model_info['model_name'] = MODEL_NAME
    model_info['dropout'] = DROPOUT
    model_info['learning_rate'] = LR
    model_info['freeze'] = FREEZE
    model_info['hidden_size'] = HIDDEN_SIZE
    model_info['batch_norm'] = BATCH_NORM
    
    if CHECKPOINT is not None:
        model_info['checkpoint'] = CHECKPOINT
        
    trainer = Trainer(json_info=model_info)
    
    print(f"===== START EVALUATE KFOLD =====")
    k_fold = 10
    scores = trainer.cross_validate(TRAIN_TXT, k_fold=k_fold)
    joblib.dump(scores, f'{SAVE_DIR}/score_kfold_{k_fold}.joblib')
    
run_kfold_validate()
    