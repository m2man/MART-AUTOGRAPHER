import os
import numpy as np
import joblib
import torch
from torch.utils.data import Dataset

class MyDatasetGenerator (Dataset):
    def __init__(self, data_dict, mode='mean'):
        self.task_id = list(data_dict.keys())
        self.task_lbl = []
        self.task_ft = []

        if mode == 'mean': # take mean of all images as 1 sample
            for task in self.task_id:
                self.task_ft.append(np.mean(data_dict[task]['features'], axis=0))
                act = task.split('_')[-1] # act01
                lbl = np.array([int(act[-2:])-1]) # convert act into index act01 --> 0
                self.task_lbl.append(lbl)

        if mode == 'none': # each image will be a sample
            for task in self.task_id:
                act = task.split('_')[-1] # act01
                lbl = np.array([int(act[-2:])-1]) # convert act into index act01 --> 0
                ft = data_dict[task]['features']
                for idx in ft.shape[0]:
                    self.task_lbl.append(lbl)
                    self.task_ft.append(ft[idx])

    def __getitem__(self, index):
        ft = torch.from_numpy(self.task_ft[index])
        ft = ft.type(torch.FloatTensor)
        lbl = torch.LongTensor(self.task_lbl[index])
        return ft, lbl
  
    def __len__(self):
        return len(self.task_lbl)