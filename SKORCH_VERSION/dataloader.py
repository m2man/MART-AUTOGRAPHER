import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

class MyDatasetGenerator (Dataset):
    
    #-------------------------------------------------------------------------------- 
    
    def __init__ (self, pathImageDirectory, pathDatasetFile, transform):
    
        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transform
    
        #---- Open file, get image paths and labels
    
        fileDescriptor = open(pathDatasetFile, "r")
        
        #---- get into the loop
        line = True
        
        while line:
                
            line = fileDescriptor.readline()
            
            #--- if not empty
            if line:
          
                lineItems = line.split(' ')
                fileName = lineItems[0]
                imagePath = os.path.join(pathImageDirectory, fileName)
                imageLabel = int(lineItems[1])-1 # original label 1 -> 20 ==> should change to 0 -> 19
                # imageLabel = imageLabel.astype(np.int64)
                
                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)   
            
        fileDescriptor.close()
    
    #-------------------------------------------------------------------------------- 
    
    def __getitem__(self, index):
        
        imagePath = self.listImagePaths[index]
        
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel= self.listImageLabels[index]
        
        if self.transform != None: imageData = self.transform(imageData)
        
        return imageData, imageLabel
        
    #-------------------------------------------------------------------------------- 
    
    def __len__(self):
        
        return len(self.listImagePaths)


# ========================================================================== #
class MyDataset_CV (Dataset):
    
    #-------------------------------------------------------------------------------- 
    
    def __init__ (self, pathImageDirectory, list_line, transform):
    
        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transform
        
        #---- get into the loop
        
        for line in list_line:
            
            #--- if not empty
            if len(line) > 0:
          
                lineItems = line.split(' ')
                fileName = lineItems[0]
                imagePath = os.path.join(pathImageDirectory, fileName)
                imageLabel = int(lineItems[1])-1 # original label 1 -> 20 ==> should change to 0 -> 19
                # imageLabel = imageLabel.astype(np.int64)
                
                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)   

    #-------------------------------------------------------------------------------- 
    
    def __getitem__(self, index):
        
        imagePath = self.listImagePaths[index]
        
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel= self.listImageLabels[index]
        
        if self.transform != None: imageData = self.transform(imageData)
        
        return imageData, imageLabel
        
    #-------------------------------------------------------------------------------- 
    
    def __len__(self):
        
        return len(self.listImagePaths)