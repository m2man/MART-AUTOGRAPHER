import os
import numpy as np
import joblib
import torch
from torch.utils.data import Dataset

class MyDatasetGenerator (Dataset):
    def __init__(self, df):
        # input is a pandas dataframe
        # last column is the label
        self.lbl = []
        self.ft = []
        
        for row in range(len(df)):
            data = df.iloc[row, :]
            ft = data[0:-1]
            self.ft.append(ft)
            
            lbl = int(data[-1])-1
            lbl = np.array([int(act[-2:])-1])
            self.lbl.append(lbl)

    def __getitem__(self, index):
        ft = torch.from_numpy(self.ft[index])
        ft = ft.type(torch.FloatTensor)
        lbl = torch.LongTensor(self.lbl[index])
        return ft, lbl
  
    def __len__(self):
        return len(self.lbl)