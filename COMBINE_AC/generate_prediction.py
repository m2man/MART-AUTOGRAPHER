from controller import Evaluator
from PIL import Image
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import joblib

PANDAS_DIR = '/mnt/sda/hong01-data/MART_DATA/OUTPUT_MERGED/PANDAS'
IMG_DIR = '/mnt/sda/hong01-data/MART_DATA/OUTPUT_MERGED/AUTOGRAPHER'
all_img_files = sorted(os.listdir(IMG_DIR))

data = pd.read_csv(f"{PANDAS_DIR}/test.csv")
sub_id_csv = list(data['sub_id'])
event_id_csv = list(data['event_id'])
task_id_csv = [f"{x}_{y}" for x, y in zip(sub_id_csv, event_id_csv)]


a = joblib.load('../DUYEN_JOBLIB/tabular_with_images_test.joblib')
# a = a.drop(columns=['event_id', 'source', 'sub_id', 'img_order'])
if 'label' in list(a.columns):
    a = a.drop(columns=['label'])
    
TABULAR_INPUT_DIM = len(a.columns) - 4 - 1 # last column is the label
TABULAR_LAYER_DIM = [1000, 500, 128]
IMG_HIDDEN_SIZE = 512

IMG_BACKBONE = 'resnet' # or resnet34
# CHECKPOINT = None 
CHECKPOINT = 'RUN_2_Unfreeze/COMBINE-R34-27082020-135631.pth.tar'
DROPOUT = 0.5
BATCH_NORM = True
CROP_SIZE = 224 # DO NOT CHANGE THIS
    
model_info = {}
model_info['crop_size'] = CROP_SIZE
model_info['checkpoint'] = CHECKPOINT
model_info['img_hidden_size'] = IMG_HIDDEN_SIZE
model_info['batch_norm'] = BATCH_NORM
model_info['dropout'] = DROPOUT
model_info['tabular_input_dim'] = TABULAR_INPUT_DIM
model_info['tabular_layer_dim'] = TABULAR_LAYER_DIM
model_info['img_backbone'] = IMG_BACKBONE
evaluator = Evaluator(json_info=model_info)

print('Predicting ...')
result = np.zeros((len(task_id_csv), 20))
for idx_task, task in tqdm(enumerate(task_id_csv)):
    sub_id, event_id = task.split('_')
    b = a[a.sub_id.eq(int(sub_id)) & a.event_id.eq(event_id)]
    b = b.drop(columns=['event_id', 'source', 'sub_id', 'img_order'])
    task_probs = np.zeros((len(b), 20))
    
    for idx_row in range(len(b)):
        data = b.iloc[idx_row,0:-1]
        data = np.array(list(data), dtype=np.float)
        
        img_path = str(b.iloc[idx_row,-1])
        img_path = '_'.join(img_path.split('_test_'))
        img_path = f"{IMG_DIR}/{img_path}"

        pred_lbl, probs_lbl = evaluator.get_pred(img_path, data)
        
        task_probs[idx_row] = probs_lbl
        
    task_mean_probs = np.mean(task_probs, axis=0)    
    result[idx_task] = task_mean_probs
    
joblib.dump(result, 'tabular_r34_prediction.joblib')