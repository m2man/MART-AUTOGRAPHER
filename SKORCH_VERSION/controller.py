from model import EfficientNet_MART
from dataloader import MyDatasetGenerator, MyDataset_CV
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torch
import time
from tensorboardX import SummaryWriter
import numpy as np
from PIL import Image
import os
device = torch.device('cuda')
import random

from sklearn.model_selection import train_test_split
from skorch.helper import predefined_split
from skorch.dataset import Dataset
from skorch import NeuralNetClassifier
from skorch.callbacks import LRScheduler, Checkpoint, EpochScoring, EarlyStopping, TensorBoard, Callback
from sklearn.model_selection import cross_validate

class Trainer():
    def __init__(self, json_info):
        super(Trainer, self).__init__()
        # ========== INIT INFORMATION ==========
        self.model_name = json_info['model_name']
        self.batch_size = json_info['batch_size']
        self.crop_size = json_info['crop_size']
        self.max_epoch = json_info['max_epoch']
        self.data_dir = json_info['data_dir']
        self.train_txt = json_info['train_txt']
        self.val_txt = json_info['val_txt']
        
        try:
            self.hidden_size = json_info['hidden_size'] # Mostly is NONE
        except:
            self.hidden_size = 1024
        
        try:
            self.freeze = json_info['freeze'] # Mostly is NONE
        except:
            self.freeze = True
        
        try:
            self.batch_norm = json_info['batch_norm'] # Mostly is NONE
        except:
            self.batch_norm = False
        
        # try:
        #     self.test_txt = json_info['test_txt'] # Mostly is NONE
        # except:
        #     self.test_txt = None
            
        self.save_dir = json_info['save_dir']
        self.optimizer = json_info['optimizer']
        self.learning_rate = json_info['learning_rate']
        
        try:
            self.checkpoint = json_info['checkpoint'] # Mostly is NONE
        except:
            self.checkpoint = None
            
        try:
            self.dropout = json_info['dropout']
        except:
            self.dropout = None
            
        # ========== SETTINGS: DEFINES MODEL ==========
        if 'efficient' in self.model_name.lower():
            structure = self.model_name.split('-')[-1].lower() # b0 or b4
            self.model = EfficientNet_MART(classCount=20, structure=structure, dropout=self.dropout, freeze=self.freeze, hidden_size = self.hidden_size, batch_norm=self.batch_norm)
            self.model = self.model.to(device)
            
            
        # ========== SETTINGS: DATA TRANSFORMS ==========
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        # transformList.append(transforms.RandomResizedCrop(CROP_SIZE))
        transformList.append(transforms.Resize(self.crop_size)) # Somehow this one is better
        transformList.append(transforms.RandomHorizontalFlip())
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)      
        self.TransformSequence=transforms.Compose(transformList)
        
        transformList = []
        transformList.append(transforms.Resize(self.crop_size))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)      
        self.TransformSequenceVal=transforms.Compose(transformList)
            
    # ---------- WRITE INFO TO TXT FILE ---------
    def extract_info(self):
        try:
            timestampLaunch = self.timestampLaunch
        except:
            timestampLaunch = 'undefined'
        model_info_log = open(f"{self.save_dir}/{self.model_name}-{timestampLaunch}-INFO.log", "w")
        model_info_log.write(f"===== {self.model_name} =====\nBATCH_SIZE: {self.batch_size}\nMAX_EPOCH: {self.max_epoch}\nDROPOUT: {self.dropout}\nLR: {self.learning_rate}\nOPTIMIZER: {self.optimizer}\nFREEZE: {self.freeze}\nHIDDEN_SIZE: {self.hidden_size}\nBATCH_NORM: {self.batch_norm}\n")
        if self.checkpoint is not None:
            model_info_log.write(f"FROM CHECKPOINT:{self.checkpoint}\n")
        model_info_log.close()
   

    # ---------- LOAD TRAINED MODEL ---------
    def load_trained_model(self):
        #---- Load checkpoint 
        if self.checkpoint is not None:
            print(f"LOAD PRETRAINED MODEL AT {self.checkpoint}")
            modelCheckpoint = torch.load(self.checkpoint)
            self.model.load_state_dict(modelCheckpoint)
        else:
            print("TRAIN FROM SCRATCH")
            
            
    # ---------- RUN TRAIN ---------
    def train(self):
        # ----- SETTINGS: LOAD PRETRAINED MODEL
        self.load_trained_model()
            
        # ----- SETTINGS: DATASET BUILDERS
        datasetTrain = MyDatasetGenerator(pathImageDirectory=self.data_dir, pathDatasetFile=self.train_txt, transform=self.TransformSequence)
        datasetVal =   MyDatasetGenerator(pathImageDirectory=self.data_dir, pathDatasetFile=self.val_txt, transform=self.TransformSequenceVal)
        
        # ----- SETTINGS: REPORT
        timestampTime = time.strftime("%H%M%S")
        timestampDate = time.strftime("%d%m%Y")
        self.timestampLaunch = timestampDate + '-' + timestampTime
        self.extract_info()
        writer = SummaryWriter(f'{self.save_dir}/{self.model_name}-{self.timestampLaunch}/')
        
        # ----- SETTINGS: CALLBACKS
        lrscheduler = LRScheduler(policy=ReduceLROnPlateau, factor = 0.2, patience=4, mode = 'min', verbose=True, min_lr=1e-5, monitor='valid_loss')
        # auc = EpochScoring(scoring='roc_auc', lower_is_better=False)
        checkpoint = Checkpoint(f_params=f"{self.save_dir}/{self.model_name}-{self.timestampLaunch}.pth.tar", monitor='valid_acc_best', f_history=f"{self.save_dir}/{self.model_name}-{self.timestampLaunch}-REPORT.json")
        earlystopping = EarlyStopping(monitor='valid_loss', patience=15)
        tensorboard = TensorBoard(writer)
        
        # ----- SETTINGS: DECLARE SKORCH
        if self.optimizer.lower() == 'adam':
            net = NeuralNetClassifier(
                self.model, 
                criterion=torch.nn.CrossEntropyLoss,
                lr=self.learning_rate,
                batch_size=self.batch_size,
                max_epochs=self.max_epoch,
                optimizer=optim.Adam,
                #optimizer__params=self.model.parameters(),
                optimizer__betas=(0.9, 0.999),
                optimizer__eps=1e-8,
                optimizer__weight_decay=1e-5,
                iterator_train__shuffle=True,
                iterator_train__num_workers=8,
                iterator_valid__shuffle=True,
                iterator_valid__num_workers=8,
                train_split=predefined_split(datasetVal),
                callbacks=[lrscheduler, checkpoint, earlystopping, tensorboard],
                device='cuda' # comment to train on cpu
            )
        if self.optimizer.lower() == 'sgd':
            net = NeuralNetClassifier(
                self.model, 
                criterion=torch.nn.CrossEntropyLoss,
                lr=self.learning_rate,
                batch_size=self.batch_size,
                max_epochs=self.max_epoch,
                optimizer=optim.SGD,
                #optimizer__params=self.model.parameters(),
                optimizer__momentum=0.9,
                optimizer__weight_decay=1e-5,
                iterator_train__shuffle=True,
                iterator_train__num_workers=8,
                iterator_valid__shuffle=True,
                iterator_valid__num_workers=8,
                train_split=predefined_split(datasetVal),
                callbacks=[lrscheduler, checkpoint, earlystopping, tensorboard],
                device='cuda' # comment to train on cpu
            )
            
        # y = np.array([y for _, y in iter(datasetTrain)])    
        net.fit(datasetTrain, y=None);
        
    # ---------- CROSS VALIDATE EVALUATION ---------
    def cross_validate(self, data_txt, k_fold=10):
        class AccuracyTracker(Callback):
            def __init__(self, track, current_run):
                self.track = track
                self.current_run = current_run
                self.max_accuracy = 0
                
            def get_track(self):
                return self.track

            def on_epoch_end(self, net, **kwargs):
                # look at the validation accuracy of the last epoch
                if net.history[-1, 'valid_acc'] >= self.max_accuracy:
                    self.max_accuracy = net.history[-1, 'valid_acc']
                    
            def on_train_end(self, net, **kwargs):
                print("~" * 60)
                print(f"MAX VALID ACC: {self.max_accuracy}")
                print("~" * 60)
                
                self.track[self.current_run] = self.max_accuracy
        
        
        structure = self.model_name.split('-')[-1].lower() # b0 or b4
        # run cross validate on data_txt file
        random.seed(1509)
        list_lines = []
        fileDescriptor = open(data_txt, "r")        
        #---- get into the loop
        line = True
        while line:
            line = fileDescriptor.readline()
            line = line.rstrip()
            if len(line) > 0:
                list_lines.append(line)
        fileDescriptor.close()
        random.shuffle(list_lines)
        numb_train = int(len(list_lines)*(k_fold-1)/k_fold)
        numb_test = len(list_lines)-numb_train
        scores = np.zeros(k_fold)
        
        for k in range(k_fold):
            print(f"========== START KFOLD: {k} ==========")
                  
            mymodel = EfficientNet_MART(classCount=20, structure=structure, dropout=self.dropout, freeze=self.freeze, hidden_size = self.hidden_size, batch_norm=self.batch_norm)
            if self.checkpoint is not None:
                print(f"LOAD PRETRAINED MODEL AT {self.checkpoint}")
                modelCheckpoint = torch.load(self.checkpoint)
                mymodel.load_state_dict(modelCheckpoint)
            
            test_list = list_lines[(k*numb_test):((k+1)*numb_test)]
            train_list = [x for x in list_lines if x not in test_list]
            datasetTrain = MyDataset_CV(pathImageDirectory=self.data_dir, list_line=train_list, transform=self.TransformSequence)
            datasetTest = MyDataset_CV(pathImageDirectory=self.data_dir, list_line=test_list, transform=self.TransformSequenceVal)
            
            # ----- SETTINGS: CALLBACKS
            lrscheduler = LRScheduler(policy=ReduceLROnPlateau, factor = 0.2, patience=4, mode = 'min', verbose=True, min_lr=1e-5, monitor='valid_loss')
            earlystopping = EarlyStopping(monitor='valid_loss', patience=15)
            accuracytracker = AccuracyTracker(track=scores, current_run=k)
            
            # ----- SETTINGS: DECLARE SKORCH
            net = NeuralNetClassifier(
                mymodel, 
                criterion=torch.nn.CrossEntropyLoss,
                lr=self.learning_rate,
                batch_size=self.batch_size,
                max_epochs=self.max_epoch,
                optimizer=optim.Adam,
                #optimizer__params=self.model.parameters(),
                optimizer__betas=(0.9, 0.999),
                optimizer__eps=1e-8,
                optimizer__weight_decay=1e-5,
                iterator_train__shuffle=True,
                iterator_train__num_workers=8,
                iterator_valid__shuffle=True,
                iterator_valid__num_workers=8,
                train_split=predefined_split(datasetTest),
                callbacks=[lrscheduler, earlystopping, accuracytracker],
                device='cuda', # comment to train on cpu
                verbose=0
            )
            
            net.fit(datasetTrain, y=None);
            
            scores = accuracytracker.get_track()
            
            print(f"KFOLD: {k} --- MaxValidAcc: {scores[k]}")
        
        return scores