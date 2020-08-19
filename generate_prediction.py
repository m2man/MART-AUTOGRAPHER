from mart_controller import MART_Evaluator
from PIL import Image
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import joblib

RUN = 'trainB' # trainA, trainB, test

IMG_DIR = '/mnt/sda/hong01-data/MART_DATA/OUTPUT_MERGED/AUTOGRAPHER'
PANDAS_DIR = '/mnt/sda/hong01-data/MART_DATA/OUTPUT_MERGED/PANDAS'

MODEL_NAME = 'EFFICIENT-B4'
CROP_SIZE = 224
CHECKPOINT = 'RUN_0_Unfreeze/EFFICIENT-B4-18082020-020739.pth.tar'
HIDDEN_SIZE = 1024
BATCH_NORM = False
    
model_info = {}
model_info['model_name'] = MODEL_NAME
model_info['crop_size'] = CROP_SIZE
model_info['checkpoint'] = CHECKPOINT
model_info['hidden_size'] = HIDDEN_SIZE
model_info['batch_norm'] = BATCH_NORM

evaluator = MART_Evaluator(model_info)

# List all images in folder
all_img_files = sorted(os.listdir(IMG_DIR))

# Extract task id
print('===== MAKING PREDICTION =====')
data = pd.read_csv(f"{PANDAS_DIR}/{RUN}.csv")
sub_id = list(data['sub_id'])
event_id = list(data['event_id'])
task_id = [f"{x}_{y}" for x, y in zip(sub_id, event_id)]

# Generate prediction scores on each task (take mean of all images in a task)
list_probs = np.zeros((len(task_id), 20))
for idx_task, task in tqdm(enumerate(task_id)):
    if RUN == 'trainA' or RUN == 'trainB':
        task = task.split('_')
        task = f"{task[0]}_{RUN}_{task[1]}"
    task_imgs = [x for x in all_img_files if task in x]
    task_probs = np.zeros((len(task_imgs), 20))
    for idx, img in enumerate(task_imgs):
        img_path = f"{IMG_DIR}/{img}"
        pred_lbl, probs_lbl = evaluator.test_image(img_path, props=True) # remember act = label+1
        task_probs[idx] = probs_lbl
    task_mean_probs = np.mean(task_probs, axis=0)
    list_probs[idx_task] = task_mean_probs
    
joblib.dump(list_probs, f'act_probs_{RUN}.joblib')