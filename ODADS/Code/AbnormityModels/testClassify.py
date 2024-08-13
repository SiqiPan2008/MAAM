import os
import torch
from torch import nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import Subset
import time
import copy
import csv
from datetime import datetime
import numpy as np
from Utils import utils
from AbnormityModels import classify
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def train(filename, device, featureExtract, modelName, numClasses, batchSize, numEpochs, LR, usePretrained, dbName, wtsName, modelType, crossValid = True):
    model, _ = abnormityModel.initializeAbnormityModel(modelName, numClasses, featureExtract, usePretrained = usePretrained)
    model = model.to(device)
    if wtsName != "":
        trainedModel = torch.load(f"ODADS/Data/Weights/{wtsName}/{wtsName}.pth")
        model.load_state_dict(trainedModel['state_dict'])  
    paramsToUpdate = model.parameters()
    print("Params to learn:")
    if featureExtract:
        paramsToUpdate = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if featureExtract:
                paramsToUpdate.append(param)
            print("\t", name)
            
    optimizer = optim.Adam(paramsToUpdate, lr = LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 7, gamma = 0.1)
    criterion = nn.NLLLoss() # what is NLL?

    model, validAccHistory, trainAccHistory, validLosses, trainLosses, LRs, timeElapsed = trainModel(device, model, criterion, optimizer, scheduler, filename, dbName, crossValid, batchSize, numEpochs, numClasses)
    
    os.makedirs(f"ODADS/Data/Results/{filename}/", exist_ok=True)
    with open(f"ODADS/Data/Results/{filename}/{filename}.csv", "w", newline="") as file:  
        writer = csv.writer(file)  
        writer.writerow(["Trained from ResNet152" if wtsName == "" else f"Trained from {wtsName}", f"Data: {dbName}", f"batchSize = {batchSize}", f"LR = {LRs[0]}", f"epochNum = {len(trainLosses)}", f"timeElapsed = {timeElapsed // 60 :.0f}m {timeElapsed % 60: .2f}s"])
        for i in range(len(trainLosses)):  
            writer.writerow([i + 1, validAccHistory[i].item(), trainAccHistory[i].item(), validLosses[i], trainLosses[i], LRs[i]])
    print(f"Data successfully written into {filename}.csv")
    
    utils.curve(validAccHistory, trainAccHistory, validLosses, trainLosses, f"ODADS/Data/Results/{filename}/{filename}.pdf")