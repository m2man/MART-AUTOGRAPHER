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
print('===== GENERATING IMAGE FEATURES =====')
data = pd.read_csv(f"{PANDAS_DIR}/{RUN}.csv")
sub_id = list(data['sub_id'])
event_id = list(data['event_id'])
task_id = [f"{x}_{y}" for x, y in zip(sub_id, event_id)]

task_ft = dict()
# loop start
for task in tqdm(task_id):
  task = task_id[0]
  if RUN == 'trainA' or RUN == 'trainB':
    task = task.split('_')
    task = f"{task[0]}_{RUN}_{task[1]}"
  task_imgs = [x for x in all_img_files if task in x]
  task_ft[task] = {}
  task_ft[task]['images'] = task_imgs
  task_imgs_ft = np.zeros((len(task_imgs), HIDDEN_SIZE))
  for idx, img in enumerate(task_imgs):
    img_path = f"{IMG_DIR}/{img}"
    img_ft = evaluator.extract_features_image(img_path)
    task_imgs_ft[idx] = img_ft.cpu().numpy()
  task_ft[task]['features'] = task_imgs_ft
# loop end
    
joblib.dump(task_ft, f'task_features_{RUN}_{CHECKPOINT.split('/')[0]}.joblib')