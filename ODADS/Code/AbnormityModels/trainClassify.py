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
from AbnormityModels import abnormityModel, testClassify
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



def trainModel(device, model, criterion, optimizer, scheduler, filename, dataDir, crossValid, batchSize, numEpochs, numClasses, usePretrained, isInception = False):
    startTime = time.time()
    bestAcc = [0]
    model.to(device)
    validAccHistory = []
    trainAccHistory = []
    validLosses = []
    trainLosses = []
    LRs = [optimizer.param_groups[0]["lr"]]
    
    dataTransforms = transforms.Compose([transforms.ToTensor()])
    if crossValid:
        imageDataset = datasets.ImageFolder(dataDir, dataTransforms)
    else:
        imageDatasets = {x: datasets.ImageFolder(os.path.join(dataDir, x), dataTransforms) for x in ["train", "valid"]}
        dataloaders = {x: torch.utils.data.DataLoader(imageDatasets[x], batch_size = batchSize, shuffle = True) for x in ["train", "valid"]}
    
    
    for epoch in range(numEpochs):
        print(f"Epoch {epoch + 1}/{numEpochs}")
        print("-" * 10)
        
        if crossValid:
            labels = np.array(imageDataset.targets)
            trainIndices = []
            validIndices = []
            for classId in range(numClasses):
                classIndices = np.where(labels == classId)[0]
                np.random.shuffle(classIndices)
                splitPoint = int(0.8 * len(classIndices))
                trainIndices.extend(classIndices[:splitPoint])
                validIndices.extend(classIndices[splitPoint:])
            trainDataset = Subset(imageDataset, trainIndices)
            validDataset = Subset(imageDataset, validIndices)
            dataloaders = {
                "train": torch.utils.data.DataLoader(trainDataset, batch_size = batchSize, shuffle=True),
                "valid": torch.utils.data.DataLoader(validDataset, batch_size = batchSize, shuffle=True)
            }
    
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            
            runningLoss = 0.0
            runningCorrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    if isInception and phase == "train":
                        outputs,auxOutputs = model(inputs)
                        outputs.to(device)
                        auxOutputs.to(device)
                        criterion.to(device)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(auxOutputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                    preds = torch.max(outputs, 1)[1]
                    if phase == "train":
                        loss.to(device)
                        loss.backward()
                        optimizer.step()
                runningLoss += loss.item() * inputs.size(0)
                runningCorrects += torch.sum(preds == labels.data)
            
            datasetLen = len(dataloaders[phase].dataset)
            epochLoss = runningLoss / datasetLen
            epochAcc = runningCorrects / datasetLen
            timeElapsed = time.time() - startTime
            print(f"time elapsed {timeElapsed // 60 :.0f}m {timeElapsed % 60 :.2f}s")
            print(f"{phase} loss: {epochLoss :.4f}, acc: {epochAcc :.4f}")
            
            if phase == "valid" and epochAcc >= bestAcc[-1]:
                bestAcc[-1] = epochAcc
                state = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict()
                }
                os.makedirs(f"ODADS/Data/Weights/{filename}/", exist_ok=True)
                epochRange = int(epoch / 10) * 10
                torch.save(state, f"ODADS/Data/Weights/{filename}/{filename} Best Epoch in {epochRange + 1} to {epochRange + 10}.pth")
                print(f"Data successfully written into {filename} Best Epoch in {epochRange + 1} to {epochRange + 10}.pth")
            if phase == "valid" and (epoch + 1) % 10 == 0:
                bestAcc.append(0)
                
            if phase == "valid":
                validAccHistory.append(epochAcc)
                validLosses.append(epochLoss)
                scheduler.step(epochLoss)
            elif phase == "train":
                trainAccHistory.append(epochAcc)
                trainLosses.append(epochLoss)
        
        print(f"optimizer learning rate: {optimizer.param_groups[0]['lr'] :.7f}")
        LRs.append(optimizer.param_groups[0]["lr"])
        print()
        
    timeElapsed = time.time() - startTime
    print(f"training complete in {timeElapsed // 60 :.0f}m {timeElapsed % 60 :.2f}s")
    print(f"best valid acc: {bestAcc}")
    
    return model, validAccHistory, trainAccHistory, validLosses, trainLosses, LRs, timeElapsed

def train(filename, device, featureExtract, modelName, numClasses, batchSize, numEpochs, LR, usePretrained, dbName, wtsName, modelType, crossValid = True):
    model, _ = abnormityModel.initializeAbnormityModel(modelName, numClasses, featureExtract, usePretrained = usePretrained)
    model = model.to(device)
    if wtsName:
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
    #criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()

    if wtsName:
        filename += " Transferred Continued" if usePretrained else " Finetuning"
    else:
        filename += " Tranferred" if usePretrained else " From Scratch"
    model, validAccHistory, trainAccHistory, validLosses, trainLosses, LRs, timeElapsed = trainModel(device, model, criterion, optimizer, scheduler, filename, dbName, crossValid, batchSize, numEpochs, numClasses, usePretrained)
    
    os.makedirs(f"ODADS/Data/Results/{filename}/", exist_ok=True)
    with open(f"ODADS/Data/Results/{filename}/{filename}.csv", "w", newline="") as file:  
        writer = csv.writer(file)  
        writer.writerow(["Trained from ResNet152" if wtsName == "" else f"Trained from {wtsName}", f"Data: {dbName}", f"batchSize = {batchSize}", f"LR = {LRs[0]}", f"epochNum = {len(trainLosses)}", f"timeElapsed = {timeElapsed // 60 :.0f}m {timeElapsed % 60: .2f}s"])
        for i in range(len(trainLosses)):  
            writer.writerow([i + 1, validAccHistory[i].item(), trainAccHistory[i].item(), validLosses[i], trainLosses[i], LRs[i]])
    print(f"Data successfully written into {filename}.csv")
    
    utils.curve(validAccHistory, trainAccHistory, validLosses, trainLosses, f"ODADS/Data/Results/{filename}/{filename}.pdf")