from mart_controller import MART_Trainer
import os

def run_train():
    BATCH_SIZE = 256
    CROP_SIZE = 224 # DO NOT CHANGE THIS
    MAX_EPOCH = 200
    DATA_DIR = '/mnt/sda/hong01-data/MART_DATA/OUTPUT_MERGED/AUTOGRAPHER'
    TRAIN_TXT = 'dataset/train.txt'
    VAL_TXT = 'dataset/val.txt'
    TEST_TXT = 'dataset/test.txt'
    CHECKPOINT = None 
    #CHECKPOINT = 'RUN_1/EFFICIENT-B4-14082020-164740.pth.tar'
    SAVE_DIR = 'RUN_2'
    OPTIM = 'SGD'
    MODEL_NAME = 'EFFICIENT-B4'
    DROPOUT = 0.5
    LR = 0.01
    
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
    model_info['test_txt'] = TEST_TXT
    model_info['save_dir'] = SAVE_DIR
    model_info['optimizer'] = OPTIM
    model_info['model_name'] = MODEL_NAME
    model_info['dropout'] = DROPOUT
    model_info['learning_rate'] = LR
    
    if CHECKPOINT is not None:
        model_info['checkpoint'] = CHECKPOINT
        
    mart_trainer = MART_Trainer(json_info=model_info)
    
    print(f"===== START TRAINING ON MART DATA =====")
    mart_trainer.train()
    
    
run_train()
    