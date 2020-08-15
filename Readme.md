# MART AUTOGRAPHER-only APPROACH
This repository is about using only autographer data to classify MART tasks. The backbone model applied in this repo is Efficientnet-B4.

## Notebook
The `Data_Process` notebook illustrated how to split dataset into train-val-test set. The result is stored in **dataset folder**. It also shown how to used the trained model to predict on a custom image or to evaluate on a train-val-test set.

## Repo structure
The **RUN folder** stores the result after training a model including: model info, report at each epoch, tensorboard folder, and trained `model.pth.tar`

`model.py`: define model

`mart_controller.py`: define training and evaluating protocol

`main.py`: train model

## Tensorboard
run the command `tensorboard --logdir= RUN_1/` to visualize the Loss, Accuracy, Learning rate of all experiments in **RUN_1 folder**

## Initial Result
Currently, we merged both trainA and trainB into 1 dataset and split train-val-test subsets based on it. The proportion was 0.8, 0.1, 0.1 for each activity. 
The model achieved 87.2% accuracy on the validate set (about 170 images) using Adam optimizer based on RUN_1_Unfreeze. We first freeze the backbone (EfficientNet) and only train the transfer layers until converge then trained all layers. Download the pretrained model [here](https://drive.google.com/file/d/1PP6erGXE9hdX3hTDCzwEvQ3e3l5XZzxD/view?usp=sharing)