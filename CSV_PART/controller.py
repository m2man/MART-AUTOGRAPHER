from model import Tabular_Model
from dataloader import MyDatasetGenerator
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torch
import time
from tensorboardX import SummaryWriter
import numpy as np
import os
device = torch.device('cuda')


class Trainer():
    def __init__(self, json_info):
        super(Trainer, self).__init__()
        # ========== INIT INFORMATION ==========
        self.layer_dim = json_info['layer_dim'] # list
        self.model_name = json_info['model_name']
        self.batch_size = json_info['batch_size']
        self.max_epoch = json_info['max_epoch']
        self.save_dir = json_info['save_dir']
        self.optimizer = json_info['optimizer']
        self.learning_rate = json_info['learning_rate']

        self.input_dim= json_info['input_dim']

        self.train_df = json_info['train_df']
        self.val_df = json_info['val_df']

        try:
            self.batch_norm = json_info['batch_norm'] # Mostly is NONE
        except:
            self.batch_norm = False

        try:
            self.checkpoint = json_info['checkpoint'] # Mostly is NONE
        except:
            self.checkpoint = None

        try:
            self.dropout = json_info['dropout']
        except:
            self.dropout = None

        # ========== SETTINGS: DEFINES MODEL ==========
        self.model = Tabular_Model(output_dim=20, input_dim=self.input_dim, dropout=self.dropout, layer_dim = self.layer_dim, batch_norm=self.batch_norm)
        self.model = self.model.to(device)

    # ---------- WRITE INFO TO TXT FILE ---------
    def extract_info(self):
        try:
            timestampLaunch = self.timestampLaunch
        except:
            timestampLaunch = 'undefined'
        model_info_log = open(f"{self.save_dir}/{self.model_name}-{timestampLaunch}-INFO.log", "w")
        model_info_log.write(f"===== {self.model_name} =====\nBATCH_SIZE: {self.batch_size}\nMAX_EPOCH: {self.max_epoch}\nDROPOUT: {self.dropout}\nLR: {self.learning_rate}\nOPTIMIZER: {self.optimizer}\nLAYER_DIM: {self.layer_dim}\nBATCH_NORM: {self.batch_norm}\nINPUT_DIM: {self.input_dim}\n")
        if self.checkpoint is not None:
            model_info_log.write(f"FROM CHECKPOINT:{self.checkpoint}\n")
        model_info_log.close()
  
    # ---------- LOAD TRAINED MODEL ---------
    def load_trained_model(self):
        #---- Load checkpoint 
        if self.checkpoint is not None:
            print(f"LOAD PRETRAINED MODEL AT {self.checkpoint}")
            modelCheckpoint = torch.load(self.checkpoint)
            self.model.load_state_dict(modelCheckpoint['state_dict'])
        else:
            print("TRAIN FROM SCRATCH")
        

    # ---------- RUN TRAIN ---------
    def train(self):
        # ----- SETTINGS: LOAD PRETRAINED MODEL
        self.load_trained_model()

        # ----- SETTINGS: DATASET BUILDERS
        datasetTrain = MyDatasetGenerator(df=self.train_df)
        datasetVal =   MyDatasetGenerator(df=self.val_df)

        dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=self.batch_size, shuffle=True,  num_workers=8, pin_memory=True)
        dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

        # ----- SETTINGS: OPTIMIZER & SCHEDULER
        if self.optimizer.lower() == 'adam':
            optimizer = optim.Adam (self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        if self.optimizer.lower() == 'sgd':
            optimizer = optim.SGD (self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor = 0.2, patience=4, mode = 'min', verbose=True, min_lr=5e-5)

        # ----- SETTINGS: LOSS FUNCTION
        loss = torch.nn.CrossEntropyLoss()

        # ----- SETTINGS: REPORT
        timestampTime = time.strftime("%H%M%S")
        timestampDate = time.strftime("%d%m%Y")
        self.timestampLaunch = timestampDate + '-' + timestampTime
        f_log = open(f"{self.save_dir}/{self.model_name}-{self.timestampLaunch}-REPORT.log", "w")
        writer = SummaryWriter(f'{self.save_dir}/{self.model_name}-{self.timestampLaunch}/')
        self.extract_info()
    
        # ---- TRAIN THE NETWORK
        lossMIN = 100000
        accMax = 0
        flag = 0
        count_change_loss = 0

        for epochID in range (0, self.max_epoch):

            if epochID > int(self.max_epoch/2) and flag == 0:
                flag = 1
                optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
                scheduler = ReduceLROnPlateau(optimizer, factor = 0.5, patience=3, mode = 'min', verbose=True, min_lr=1e-5)

            print(f"Training {epochID}/{self.max_epoch-1}")

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime

            lossTrain = self.epochTrain(dataLoaderTrain, optimizer, loss)
            lossVal, acc = self.epochVal(dataLoaderVal, loss)

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime

            scheduler.step(lossVal+2*(1-acc))

            if lossVal < lossMIN or acc > accMax:
                count_change_loss = 0
                if lossVal < lossMIN:
                    lossMIN = lossVal
                if acc > accMax:
                    accMax = acc
                torch.save({'epoch': epochID + 1, 'state_dict': self.model.state_dict(), 'best_loss': lossMIN, 'best_acc': accMax}, f"{self.save_dir}/{self.model_name}-{self.timestampLaunch}.pth.tar")
                print ('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] lossVal= ' + str(lossVal) + ' --- ACC: ' + str(acc))
                f_log.write(f"[{timestampEND} - {epochID+1} - SAVE]\nLoss_Train: {lossTrain}\nLoss_Val: {lossVal}\nACC_Val: {acc}\n")
                loss_val_save = lossVal
                loss_train_save = lossTrain
                acc_val_save = acc
            else:
                count_change_loss += 1
                print ('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] lossVal= ' + str(lossVal) + ' --- ACC: ' + str(acc))
                f_log.write(f"[{timestampEND} - {epochID+1}]\nLoss_Train: {lossTrain}\nLoss_Val: {lossVal}\nACC_Val: {acc}\n")

            writer.add_scalars('Loss', {'train': lossTrain}, epochID)
            writer.add_scalars('Loss', {'val': lossVal}, epochID)
            writer.add_scalars('Loss', {'val-best': lossMIN}, epochID)

            writer.add_scalars('Accuracy', {'val': acc}, epochID)
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning Rate', current_lr, epochID)

            if count_change_loss >= 25:
                print(f'Early stopping: {count_change_loss} epoch not decrease the loss')
                break

        f_log.close()
        writer.close()
        
        output = dict()
        output['loss_val_save'] = loss_val_save
        output['loss_train_save'] = loss_train_save
        output['acc_val_save'] = acc_val_save
        
        return output
    # ---------- 1-EPOCH TRAIN ---------
    def epochTrain(self, dataLoader, optimizer, loss):
      
        self.model.train()
        loss_report = 0
        count = 0

        for batchID, (input, target) in enumerate (dataLoader): 

            # target = target.cuda(async = True)
            target = target.flatten() # [0, 2, 18, 1, ...]
            target = target.to(device)

            varInput = input.to(device)
            varOutput = self.model(varInput)
            lossvalue = loss(varOutput, target)

            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()

            loss_report += lossvalue.item()
            count += 1

            if (batchID+1) % 100 == 0:
                print(f"Batch Idx: {batchID+1} / {len(dataLoader)}: Loss Train {loss_report/count}")

        return loss_report/count
    
    # ---------- VALIDATE ---------
    def epochVal (self, dataLoader, loss):
  
        self.model.eval()

        outGT = torch.LongTensor().to(device)
        outPRED = torch.LongTensor().to(device)

        lossVal = 0
        lossValNorm = 0
        #losstensorMean = 0

        with torch.no_grad():
            for i, (input, target) in enumerate (dataLoader):

                # target = target.cuda(async=True)
                target = target.flatten()
                target = target.to(device)

                varInput = input.to(device) 

                varOutput = self.model(varInput)
                varOutput_softmax = torch.nn.functional.log_softmax(varOutput, dim=1)
                # varOutput_softmax = torch.nn.functional.softmax(varOutput, dim=1)
                varOutput_class = torch.max(varOutput_softmax, dim=1).indices

                losstensor = loss(varOutput, target)

                lossVal += losstensor.item()
                lossValNorm += 1

                outGT = torch.cat((outGT, target), 0)
                outPRED = torch.cat((outPRED, varOutput_class), 0)
    
        # print(f'GT shape: {outGT.shape} - PRED shape: {outPRED.shape}')
        outLoss = lossVal / lossValNorm
        accuracy = torch.sum(outGT==outPRED).item() / outGT.size()[0]
        # losstensorMean = losstensorMean / lossValNorm

        return outLoss, accuracy
        

class Evaluator():
    def __init__(self, json_info):
        super(Evaluator, self).__init__()
        # ========== INIT INFORMATION ==========
        # self.model_name = json_info['model_name']
        self.input_dim= json_info['input_dim']

        try:
            self.checkpoint = json_info['checkpoint'] # Mostly is NONE
        except:
            self.checkpoint = None
        
        self.layer_dim = json_info['layer_dim'] # Mostly is NONE
  
        try:
            self.batch_norm = json_info['batch_norm'] # Mostly is NONE
        except:
            self.batch_norm = False
        
        try:
            self.dropout = json_info['dropout']
        except:
            self.dropout = None
    
        # ========== SETTINGS: DEFINES MODEL ==========
        self.model = Tabular_Model(output_dim=20, input_dim=self.input_dim, dropout=self.dropout, layer_dim = self.layer_dim, batch_norm=self.batch_norm)
        self.model = self.model.to(device)
        self.load_trained_model()
        self.model.eval()
  
    # ---------- LOAD TRAINED MODEL ---------
    def load_trained_model(self):
        #---- Load checkpoint 
        if self.checkpoint is not None:
            print(f"LOAD PRETRAINED MODEL AT {self.checkpoint}")
            modelCheckpoint = torch.load(self.checkpoint)
            self.model.load_state_dict(modelCheckpoint['state_dict'])
        else:
            print("NO PRETRAINED PROVIDED")
          
    # ---------- EVALUATE LIST OF IMAGES FROM TXT FILE ---------
    def evaluate(self, data_df):
        # ----- SETTINGS: DATASET BUILDERS
        datasetVal =   MyDatasetGenerator(df=data_df)
        dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=4, shuffle=False, num_workers=8, pin_memory=True)

        outGT = torch.LongTensor().to(device)
        outPRED = torch.LongTensor().to(device)

        # ----- SETTINGS: LOSS FUNCTION
        loss = torch.nn.CrossEntropyLoss()
        lossVal = 0
        lossValNorm = 0
        #losstensorMean = 0
        
        # self.model.eval()
        with torch.no_grad():
            for i, (input, target) in enumerate (dataLoaderVal):
                # target = target.cuda(async=True)
                target = target.flatten()
                target = target.to(device)

                varInput = input.to(device)   

                varOutput = self.model(varInput)
                varOutput_softmax = torch.nn.functional.log_softmax(varOutput, dim=1)
                varOutput_class = torch.max(varOutput_softmax, dim=1).indices

                losstensor = loss(varOutput, target)

                lossVal += losstensor.item()
                lossValNorm += 1

                outGT = torch.cat((outGT, target), 0)
                outPRED = torch.cat((outPRED, varOutput_class), 0)

        # print(f'GT shape: {outGT.shape} - PRED shape: {outPRED.shape}')
        outLoss = lossVal / lossValNorm
        accuracy = torch.sum(outGT==outPRED).item() / outGT.size()[0]
        # losstensorMean = losstensorMean / lossValNorm

        return outLoss, accuracy
  
    # ---------- PREDICT ON CUSTOM TASK_ID (ImageFT) ---------
    def test_task(self, inputs, props=True):
        task_ft = torch.from_numpy(inputs)
        task_ft = task_ft.type(torch.FloatTensor)
        task_ft = task_ft.unsqueeze(0)
        # varInput = torch.autograd.Variable(imageData).to(device)
        # self.model.eval()
        with torch.no_grad():
            varInput = task_ft.to(device)
            varOutput = self.model(varInput)
            if props:
                varOutput_softmax = torch.nn.functional.softmax(varOutput, dim=1)
            else:
                varOutput_softmax = torch.nn.functional.log_softmax(varOutput, dim=1)
            varOutput_class = torch.max(varOutput_softmax, dim=1).indices

            pred_lbl = varOutput_class.item()
            probs_lbl = varOutput_softmax.cpu().numpy()
            probs_lbl = np.squeeze(probs_lbl)
        
        return pred_lbl, probs_lbl
    
    # ---------- EXTRACT FEATURE ON CUSTOM IMAGE ---------
    def extract_features(self, row_ft):
        # row_ft is the standardized numpy array
        # self.model.eval()
        task_ft = torch.from_numpy(row_ft)
        task_ft = task_ft.type(torch.FloatTensor)
        task_ft = task_ft.unsqueeze(0)
        with torch.no_grad():
            varInput = task_ft.to(device)
            ft = self.model.extract_features(inputs=varInput, train=False)
        return ft