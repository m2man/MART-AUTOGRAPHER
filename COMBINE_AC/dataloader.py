import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

DATA_DIR = '/mnt/sda/hong01-data/MART_DATA/OUTPUT_MERGED/AUTOGRAPHER'

class MyDatasetGenerator (Dataset):
    def __init__(self, df, transform=None):
        # input is a pandas dataframe
        # second last column is img_path
        # last column is the label
        self.lbl = []
        self.img_path = []
        self.tab = []
        self.transform = transform
        
        for row in range(len(df)):
            data = df.iloc[row,0:-2]
            data = np.array(list(data), dtype=np.float)
            self.tab.append(data)
            
            img_path = str(df.iloc[row,-2])
            img_path = f"{DATA_DIR}/{img_path}"
            self.img_path.append(img_path)
            
            lbl = int(df.iloc[row,-1])-1
            lbl = np.array([lbl])
            self.lbl.append(lbl)

    def __getitem__(self, index):
        img_path = self.img_path[index]
        img_ft = Image.open(img_path).convert('RGB')
        
        tab_ft = torch.from_numpy(self.tab[index])
        tab_ft = tab_ft.type(torch.FloatTensor)
        
        lbl = torch.LongTensor(self.lbl[index])
        
        if self.transform != None: img_ft = self.transform(img_ft)
            
        return img_ft, tab_ft, lbl
  
    def __len__(self):
        return len(self.lbl)