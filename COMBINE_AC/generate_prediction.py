from controller import Evaluator
from PIL import Image
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import joblib

a = joblib.load('../DUYEN_JOBLIB/tabular_with_images_test.joblib')
event_id = list(a['event_id'])
source = list(a['source'])
sub_id = list(a['sub_id'])
a = a.drop(columns=['event_id', 'source', 'sub_id'])
if 'label' in list(a.columns):
    a = a.drop(columns=['label'])

    
TABULAR_INPUT_DIM = len(train_df.columns) - 2 # last column is the label
TABULAR_LAYER_DIM = [1000, 500, 128]
IMG_HIDDEN_SIZE = 512

IMG_BACKBONE = 'efficient' # or resnet34
# CHECKPOINT = None 
CHECKPOINT = 'RUN_1/COMBINE-E4-27082020-114415.pth.tar'
DROPOUT = 0.5
LR = 0.001
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