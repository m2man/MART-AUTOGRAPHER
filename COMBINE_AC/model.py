import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

device = torch.device('cuda')

# ---------------- AUTOGRAPHER PART ----------------
class Autographer_E4(nn.Module):
    def __init__(self, structure='b4', freeze=True, dropout=None, hidden_size=1024, batch_norm=False):
        super(Autographer_E4, self).__init__()
        # Init model
        self.structure = structure
        self.dropout = dropout
        full_structure = 'efficientnet-' + self.structure
        self.backbone = EfficientNet.from_pretrained(full_structure)
        self.hidden_size = hidden_size
        
        # freeze backbone
        if freeze:
            for n, p in self.backbone.named_parameters():
                p.requires_grad = False
        
        if batch_norm:
            self.bn = nn.BatchNorm1d(num_features=self.hidden_size)
        else:
            self.bn = None
            
        # Final linear layer
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        if structure == 'b4':
            features_shape = 1792
        if structure == 'b0':
            features_shape = 1280
        self.fc = nn.Linear(features_shape, self.hidden_size)
        if self.dropout:
            self.dropout = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, inputs):
        x = self.backbone.extract_features(inputs)
        x = self.avg_pooling(x)
        x = x.flatten(start_dim=1)
        if self.dropout:
            x = self.dropout(x)           
        x = self.fc(x)
        if self.bn is not None:
            x = self.bn(x)
        x = F.relu(x)
        if self.dropout is not None:
            x = self.dropout2(x)
        return x

    

class Autographer_ResNet34(nn.Module):
    def __init__(self, freeze=True, dropout=None, hidden_size=1024, batch_norm=False):
        super(Autographer_ResNet34, self).__init__()
        # Init model
        self.dropout = dropout
        self.backbone = models.resnet34(pretrained=True)
        self.hidden_size = hidden_size
        
        # freeze backbone
        if freeze:
            ct = 0
            for child in self.backbone.children():
                ct += 1
                if ct < 10:
                    for param in child.parameters():
                        param.requires_grad = False
        
        if batch_norm:
            self.bn = nn.BatchNorm1d(num_features=self.hidden_size)
        else:
            self.bn = None
            
        # Final linear layer
        features_shape = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(features_shape, self.hidden_size)
        if self.dropout is not None:
            self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs):
        x = self.backbone(inputs)
        if self.bn is not None:
            x = self.bn(x)
        x = F.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x
    
    

# ---------------- TABULAR PART ----------------    
class Tabular(nn.Module):
    def __init__(self, input_dim=102, layer_dim=[1000, 500, 128], dropout=0.5, batch_norm=True):
        super(Tabular, self).__init__()
        # Init model
        self.input_dim = input_dim
        self.batch_norm = batch_norm
        self.dropout = dropout
        
        if isinstance(layer_dim, list):
            self.layer_dim = layer_dim
        else:
            self.layer_dim = [layer_dim]
             
        if dropout is not None:
            self.DropoutList = nn.ModuleList([nn.Dropout(dropout) for _ in range(len(self.layer_dim))])
            
        if batch_norm:
            self.BatchnormList = nn.ModuleList([nn.BatchNorm1d(num_features=x) for x in self.layer_dim])
            self.firstbn = nn.BatchNorm1d(num_features=self.input_dim)
        
        self.LinearList = nn.ModuleList([nn.Linear(self.input_dim, self.layer_dim[0])])
        self.LinearList.extend([nn.Linear(self.layer_dim[x], self.layer_dim[x+1]) for x in range(0, len(self.layer_dim)-1)])
  
    def forward(self, inputs):
        x = self.firstbn(inputs)
        for step in range(len(self.layer_dim)):
            x = self.LinearList[step](x)
            if self.batch_norm:
                x = self.BatchnormList[step](x)
            x = F.relu(x)
            if self.dropout is not None:
                x = self.DropoutList[step](x)     
        return x

    
    
# ---------------- COMBINE ---------------- 
class Combine_Model(nn.Module):
    def __init__(self, tabular_input_dim=102, tabular_layer_dim=[1000, 500, 128], img_freeze=True, img_hidden_size=512, img_backbone = 'efficient', dropout=0.5, batch_norm=True):
        super(Combine_Model, self).__init__()
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.img_backbone = img_backbone
        
        if self.img_backbone.lower() == 'efficient':
            print("USING EFFICIENTNET")
            self.img = Autographer_E4(structure='b4', freeze=img_freeze, dropout=dropout, hidden_size=img_hidden_size, batch_norm=batch_norm)
        else: # resnet (ResNet34)
            print("USING RESNET34")
            self.img = Autographer_ResNet34(freeze=img_freeze, dropout=dropout, hidden_size=img_hidden_size, batch_norm=batch_norm)
            
        self.tabular = Tabular(input_dim=tabular_input_dim, layer_dim=tabular_layer_dim, dropout=dropout, batch_norm=batch_norm)
        self.merge = nn.Linear(img_hidden_size + tabular_layer_dim[-1], 128)
        self.classifier = nn.Linear(128, 20)
        
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(num_features=img_hidden_size + tabular_layer_dim[-1])
            self.bn2 = nn.BatchNorm1d(num_features=128)
        if self.dropout is not None:
            self.do = nn.Dropout(dropout)
        
    def forward(self, img, tabular):
        img_ft = self.img(img)
        tab_ft = self.tabular(tabular)
        cat_ft = torch.cat([img_ft, tab_ft], dim=1)
        if self.batch_norm:
            cat_ft = self.bn1(cat_ft)
        x = self.merge(cat_ft)
        if self.batch_norm:
            x = self.bn2(x)
        x = F.relu(x)
        if self.dropout is not None:
            x = self.do(x)
        x = self.classifier(x)
        return x
    