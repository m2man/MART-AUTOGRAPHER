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
CHECKPOINT = 'RUNS/RUN_5_Unfreeze/EFFICIENT-B4-20082020-121209.pth.tar'
HIDDEN_SIZE = 512
BATCH_NORM = True
DROPOUT = 0.5
    
model_info = {}
model_info['model_name'] = MODEL_NAME
model_info['crop_size'] = CROP_SIZE
model_info['checkpoint'] = CHECKPOINT
model_info['hidden_size'] = HIDDEN_SIZE
model_info['batch_norm'] = BATCH_NORM
model_info['dropout'] = DROPOUT

evaluator = MART_Evaluator(model_info)

# List all images in folder
all_img_files = sorted(os.listdir(IMG_DIR))

# Extract task id
print('===== GENERATING IMAGE FEATURES =====')
data = pd.read_csv(f"{PANDAS_DIR}/{RUN}.csv")
sub_id = list(data['sub_id'])
source = list(data['source'])
event_id = list(data['event_id'])
task_id = [f"{x}_{y}_{z}" for x, y, z in zip(sub_id, source, event_id)]

task_ft = dict()
# loop start
for task in tqdm(task_id):
    if RUN == 'test':
        temp = task.split('_')
        task = '_'.join([temp[0], temp[2]])
    task_imgs = [x for x in all_img_files if task in x]
    #print(task_imgs)
    task_ft[task] = {}
    task_ft[task]['images'] = task_imgs
    task_imgs_ft = np.zeros((len(task_imgs), HIDDEN_SIZE))
    for idx, img in enumerate(task_imgs):
        img_path = f"{IMG_DIR}/{img}"
        img_ft = evaluator.extract_features_image(img_path)
        task_imgs_ft[idx] = img_ft.cpu().numpy()
    task_ft[task]['features'] = task_imgs_ft
# loop end
    
joblib.dump(task_ft, f"../joblib_files/autographer_embeded_ft_{RUN}_{CHECKPOINT.split('/')[1]}.joblib")