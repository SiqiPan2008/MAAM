import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models, datasets
from torch.utils.data import Subset, Dataloader
import imageio
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image
import csv
from datetime import datetime
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def resizeLongEdge(img, longEdgeSize = 224):
    width, height = img.size
    if width > height:
        newSize = (longEdgeSize, int(height * longEdgeSize / width))
        loc = (0, int((224 - newSize[1]) / 2))
    else:
        newSize = (int(longEdgeSize * width / height), longEdgeSize)
        width, _ = img.size
        loc = (int((224 - newSize[0]) / 2), 0)
    img = img.resize(newSize)
    blackBackground = Image.new("RGB", (224, 224), "black")  
    blackBackground.paste(img, loc)
    return blackBackground

def setParameterRequiresGrad(model, featureExtract):
    if featureExtract:
        for param in model.parameters():
            param.requires_grad = False

def initializeModel (modelName, numClasses, featureExtract, usePretrained = True):
    modelFt = None
    inputSize = 0
    
    if modelName == "resnet":
        modelFt = models.resnet152(weights = models.ResNet152_Weights.DEFAULT if usePretrained else None)
        setParameterRequiresGrad(modelFt, featureExtract if usePretrained else False)
        numFtrs = modelFt.fc.in_features
        modelFt.fc = nn.Sequential(nn.Linear(numFtrs, numClasses), nn.LogSoftmax(dim = 1))
        inputSize = 224
    else:
        print("Invalid model name.")
        exit()
    
    return modelFt, inputSize

def curve(validAccHistory, trainAccHistory, validLosses, trainLosses, filename):
    x = range(1, len(trainLosses) + 1)
    
    vAH = [validAccHistory[i].item() for i in range(len(x))]
    tAH = [trainAccHistory[i].item() for i in range(len(x))]
    plt.subplot(2, 1, 1)
    plt.plot(x, vAH, label = "validAcc")
    plt.plot(x, tAH, label = "trainAcc")
    plt.title("Accuracy Curve")  
    plt.xlabel("Epoch")  
    plt.ylabel("Accuracy")  
    plt.legend()  
      
    plt.subplot(2, 1, 2)
    plt.plot(x, validLosses, label = "validLoss")
    plt.plot(x, trainLosses, label = "trainLoss")
    plt.title("Loss Curve")  
    plt.xlabel("Epoch")  
    plt.ylabel("Loss")  
    plt.legend() 
    
    plt.tight_layout()
    plt.savefig(os.path.join(".\\Log", filename), format="pdf")
    # plt.show()

def divideDataset(fullDataset, crossValidSegs):  
    indices = list(range(len(fullDataset)))  
    random.shuffle(indices)  
    
    dataLen = len(fullDataset)
    segSize = int(dataLen // crossValidSegs)
    segSeps = list(range(0, dataLen, segSize))
    segSeps[crossValidSegs] = dataLen
    segIndices = [indices[segSeps[i] : segSeps[i + 1]] for i in range(crossValidSegs)]
    return segIndices
      
    # 使用 Subset 创建训练和验证集  
    train_dataset = Subset(full_dataset, train_indices)  
    val_dataset = Subset(full_dataset, val_indices)  
      
    return train_dataset, val_dataset 

def genCrossValidDataloader(fullDataset, segIndices, validIndex, batchSize):
    trainIndices = []
    validIndices = segIndices[validIndex]
    for i in range(len(segIndices)):
        if i != validIndex:
            trainIndices += segIndices[i]
    datasets = {"train": Subset(fullDataset, trainIndices),
        "valid": Subset(fullDataset, validIndices)}
    return {x: torch.utils.data.Dataloader(datasets[x], batch_size = batchSize, shuffle = True) for x in ["train", "valid"]}

def trainModelWithCrossValid(device, model, segIndices, imageDataset, criterion, optimizer, scheduler, filename, dbName, crossValid, batchSize, numEpochs, isInception = False):
    startTime = time.time()
    bestAcc = 0
    model.to(device)
    validAccHistory = []
    trainAccHistory = []
    validLosses = []
    trainLosses = []
    LRs = [optimizer.param_groups[0]["lr"]]
    
    bestModelWts = copy.deepcopy(model.state_dict())
    
    for epoch in range(numEpochs):
        print(f"Epoch {epoch + 1}/{numEpochs}")
        print("-" * 10)
        
        dataloaders = genCrossValidDataloader(imageDataset, segIndices, random.randint(0, len(segIndices) - 1), batchSize)
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train() #看下定义
            else:
                model.eval() #看下定义
            
            runningLoss = 0.0
            runningCorrects = 0
            
            for inputs, labels in dataloaders:
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
                        
                    preds = torch.max(outputs, 1)[1] # what does this function mean?
                    if phase == "train":
                        loss.to(device)
                        loss.backward()
                        optimizer.step()
                runningLoss += loss.item() * inputs.size(0) # what does 0 mean
                runningCorrects += torch.sum(preds == labels.data)
            
            datasetLen = len(dataloaders[phase].dataset)
            epochLoss = runningLoss / datasetLen
            epochAcc = runningCorrects / datasetLen
            timeElapsed = time.time() - startTime
            print(f"time elapsed {timeElapsed // 60 :.0f}m {timeElapsed % 60 :.2f}s")
            print(f"{phase} loss: {epochLoss :.4f}, acc: {epochAcc :.4f}")
            
            if phase == "valid" and epochAcc >= bestAcc:
                bestAcc = epochAcc
                bestModelWts = copy.deepcopy(model.state_dict())
                state = {
                    "state_dict": model.state_dict(),
                    "best_acc": bestAcc,
                    "optimizer": optimizer.state_dict()
                }
                torch.save(state, os.path.join(".\\TrainedModel", filename))
                print(f"Data successfully written into {filename}")
                
            if phase == "valid":
                validAccHistory.append(epochAcc)
                validLosses.append(epochLoss)
                # scheduler.step(epochLoss)
            elif phase == "train":
                trainAccHistory.append(epochAcc)
                trainLosses.append(epochLoss)
        
        print(f"optimizer learning rate: {optimizer.param_groups[0]['lr'] :.7f}")
        LRs.append(optimizer.param_groups[0]["lr"])
        print()
        
    timeElapsed = time.time() - startTime
    print(f"training complete in {timeElapsed // 60 :.0f}m {timeElapsed % 60 :.2f}s")
    print(f"best valid acc: {bestAcc :.4f}")
    model.load_state_dict(bestModelWts)
    
    return model, validAccHistory, trainAccHistory, validLosses, trainLosses, LRs, timeElapsed

def trainModel(device, model, dataloaders, criterion, optimizer, scheduler, filename, dbName, crossValid, numEpochs, isInception = False):
    startTime = time.time()
    bestAcc = 0
    model.to(device)
    validAccHistory = []
    trainAccHistory = []
    validLosses = []
    trainLosses = []
    LRs = [optimizer.param_groups[0]["lr"]]
    
    bestModelWts = copy.deepcopy(model.state_dict())
    
    for epoch in range(numEpochs):
        print(f"Epoch {epoch + 1}/{numEpochs}")
        print("-" * 10)
        
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train() #看下定义
            else:
                model.eval() #看下定义
            
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
                        
                    preds = torch.max(outputs, 1)[1] # what does this function mean?
                    if phase == "train":
                        loss.to(device)
                        loss.backward()
                        optimizer.step()
                runningLoss += loss.item() * inputs.size(0) # what does 0 mean
                runningCorrects += torch.sum(preds == labels.data)
            
            datasetLen = len(dataloaders[phase].dataset)
            epochLoss = runningLoss / datasetLen
            epochAcc = runningCorrects / datasetLen
            timeElapsed = time.time() - startTime
            print(f"time elapsed {timeElapsed // 60 :.0f}m {timeElapsed % 60 :.2f}s")
            print(f"{phase} loss: {epochLoss :.4f}, acc: {epochAcc :.4f}")
            
            if phase == "valid" and epochAcc >= bestAcc:
                bestAcc = epochAcc
                bestModelWts = copy.deepcopy(model.state_dict())
                state = {
                    "state_dict": model.state_dict(),
                    "best_acc": bestAcc,
                    "optimizer": optimizer.state_dict()
                }
                torch.save(state, os.path.join(".\\TrainedModel", filename))
                print(f"Data successfully written into {filename}")
                
            if phase == "valid":
                validAccHistory.append(epochAcc)
                validLosses.append(epochLoss)
                # scheduler.step(epochLoss)
            elif phase == "train":
                trainAccHistory.append(epochAcc)
                trainLosses.append(epochLoss)
        
        print(f"optimizer learning rate: {optimizer.param_groups[0]['lr'] :.7f}")
        LRs.append(optimizer.param_groups[0]["lr"])
        print()
        
    timeElapsed = time.time() - startTime
    print(f"training complete in {timeElapsed // 60 :.0f}m {timeElapsed % 60 :.2f}s")
    print(f"best valid acc: {bestAcc :.4f}")
    model.load_state_dict(bestModelWts)
    
    return model, validAccHistory, trainAccHistory, validLosses, trainLosses, LRs, timeElapsed

def train(device, featureExtract, modelName, numClasses, batchSize, numEpochs, LR, usePretrained, dbName, wtsName, modelType, crossValid):
    dataDir = "./Data/" + dbName
    #dataloaders = {}
    #if crossValid == 0:
    dataTransforms = transforms.Compose([transforms.Lambda(resizeLongEdge), transforms.ToTensor()])
    #imageDatasets = {x: datasets.ImageFolder(os.path.join(dataDir, x), dataTransforms) for x in ["train", "valid"]}
    #dataloaders = {x: torch.utils.data.DataLoader(imageDatasets[x], batch_size = batchSize, shuffle = True) for x in ["train", "valid"]}
    #elif crossValid == 1:
    imageDataset = datasets.ImageFolder(dataDir, dataTransforms)
    segIndices = divideDataset(imageDataset, crossValid)
    
    modelFt, inputSize = initializeModel(modelName, numClasses, featureExtract, usePretrained = usePretrained) # what does FT stand for?
    modelFt = modelFt.to(device)
    if wtsName != "":
        trainedModel = torch.load(os.path.join(".\\TrainedModel", wtsName + ".pth"))
        modelFt.load_state_dict(trainedModel['state_dict'])  
    paramsToUpdate = modelFt.parameters()
    print("Params to learn:")
    if featureExtract:
        paramsToUpdate = []
    for name, param in modelFt.named_parameters():
        if param.requires_grad:
            if featureExtract:
                paramsToUpdate.append(param)
            print("\t", name)
            
    optimizerFt = optim.Adam(paramsToUpdate, lr = LR)
    scheduler = optim.lr_scheduler.StepLR(optimizerFt, step_size = 7, gamma = 0.1)
    criterion = nn.NLLLoss() # what is NLL?
    
    now = datetime.now()
    filename = now.strftime(modelType + " %Y-%m-%d %H-%M-%S")
    
    #modelFt, validAccHistory, trainAccHistory, validLosses, trainLosses, LRs, timeElapsed = trainModel(device, modelFt, dataloaders, criterion, optimizerFt, scheduler, filename + ".pth", dbName, crossValid, numEpochs = numEpochs)
    modelFt, validAccHistory, trainAccHistory, validLosses, trainLosses, LRs, timeElapsed = trainModelWithCrossValid(device, modelFt, segIndices, imageDataset, criterion, optimizerFt, scheduler, filename + ".pth", dbName, crossValid, batchSize, numEpochs)
    with open(os.path.join(".\\Log", filename + ".csv"), "w", newline="") as file:  
        writer = csv.writer(file)  
        writer.writerow(["Trained from ResNet152" if wtsName == "" else f"Trained from {wtsName}", f"Data: {dbName}", f"batchSize = {batchSize}", f"LR = {LRs[0]}", f"epochNum = {len(trainLosses)}", f"timeElapsed = {timeElapsed // 60 :.0f}m {timeElapsed % 60: .2f}s"])
        for i in range(len(trainLosses)):  
            writer.writerow([i + 1, validAccHistory[i].item(), trainAccHistory[i].item(), validLosses[i], trainLosses[i], LRs[i]])
    print(f"Data successfully written into {filename}.csv")
    
    curve(validAccHistory, trainAccHistory, validLosses, trainLosses, filename + ".pdf")