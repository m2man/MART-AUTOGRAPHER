from controller import Evaluator
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import joblib

a = joblib.load('tabular_test.joblib')
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

print('Predicting ...')
result = np.zeros((len(a), 20))
for row in tqdm(range(len(a))):
    row_ft = a.iloc[row]
    row_ft = np.asarray(row_ft)
    pred_lbl, probs_lbl = evaluator.test_task(row_ft)
    result[row] = probs_lbl

joblib.dump(result, 'tabular_prediction.joblib')