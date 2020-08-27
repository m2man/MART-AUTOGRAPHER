from controller import Trainer
import os
import joblib
from sklearn.model_selection import train_test_split

def run_train():
    a = joblib.load('../CSV_PART/tabular_with_images_train.joblib')
    a = a.drop(columns=['event_id', 'source', 'sub_id', 'img_order'])
    train_df, val_df, y_train, y_val = train_test_split(a[list(a.columns)[:-1]], a[list(a.columns)[-1]], stratify=a[list(a.columns)[-1]], test_size=0.3, random_state=1509)
    train_df['label'] = y_train
    val_df['label'] = y_val
    
    TABULAR_INPUT_DIM = len(train_df.columns) - 2 # last column is the label
    TABULAR_LAYER_DIM = [1000, 500, 128]
    IMG_FREEZE = False
    IMG_HIDDEN_SIZE = 512
    
    TRAIN_DF = train_df
    VAL_DF = val_df
    
    MODEL_NAME = 'COMBINE-R34'
    IMG_BACKBONE = 'resnet' # or resnet34
    #CHECKPOINT = None 
    CHECKPOINT = 'RUN_2/COMBINE-R34-27082020-134115.pth.tar'
    SAVE_DIR = 'RUN_2_Unfreeze'
    OPTIM = 'Adam'
    DROPOUT = 0.5
    LR = 0.001
    BATCH_NORM = True
    CROP_SIZE = 224 # DO NOT CHANGE THIS
    MAX_EPOCH = 200
    BATCH_SIZE = 64
    
    if not os.path.exists(f'{SAVE_DIR}'):
        print(f'Creating {SAVE_DIR} folder')
        os.makedirs(f'{SAVE_DIR}')
    
    model_info = {}
    model_info['tabular_input_dim'] = TABULAR_INPUT_DIM
    model_info['tabular_layer_dim'] = TABULAR_LAYER_DIM
    model_info['img_freeze'] = IMG_FREEZE
    model_info['img_hidden_size'] = IMG_HIDDEN_SIZE
    model_info['train_df'] = TRAIN_DF
    model_info['val_df'] = VAL_DF
    model_info['model_name'] = MODEL_NAME
    model_info['optimizer'] = OPTIM
    model_info['dropout'] = DROPOUT
    model_info['learning_rate'] = LR
    model_info['batch_norm'] = BATCH_NORM
    model_info['crop_size'] = CROP_SIZE
    model_info['max_epoch'] = MAX_EPOCH
    model_info['batch_size'] = BATCH_SIZE
    model_info['save_dir'] = SAVE_DIR
    model_info['img_backbone'] = IMG_BACKBONE
    
    if CHECKPOINT is not None:
        model_info['checkpoint'] = CHECKPOINT
        
    trainer = Trainer(json_info=model_info)
    
    print(f"===== START TRAINING ON MART DATA =====")
    trainer.train()
    
    
run_train()