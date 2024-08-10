import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models, datasets
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
from Classify import classify
from TrainClassify import trainClassify
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def getCriteria():
    with open('TrainDiagnose/criteria.json', 'r', encoding='utf-8') as file:
        criteria = json.load(file)
    return criteria

class SimpleNet(nn.Module):
    def __init__(self, numClasses, dNumClasses):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(numClasses, dNumClasses)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.fc(x)
        x = self.softmax(x)
        return x

def getRandImageOutput(dbName, abnormityType, abnormity, oModel, fModel):
    pass

def getOutputAndLabel(diseaseName, oAbnormityNum, fAbnormityNum, grade, dbName):
    criteria = getCriteria()
    if oAbnormityNum == 0 or fAbnormityNum == 0:
        abnormityType = "Fundus" if oAbnormityNum == 0 else "OCT"
        diseaseAbnormities = criteria[diseaseName][abnormityType]
        allAbnormityNum = criteria["All"][abnormityType]
        selCorrectAbnormities = random.sample(diseaseAbnormities, grade)
        incorrectAbnormities = [abnormity for abnormity in allAbnormityNum if (abnormity not in diseaseAbnormities)]
        selIncorrectAbnormities = random.sample(incorrectAbnormities, fAbnormityNum - grade)
        selAbnormities = selCorrectAbnormities + selIncorrectAbnormities
        output = torch.empty([fAbnormityNum, allAbnormityNum])
        for abnormity in selAbnormities:
            output[abnormity] = getRandImageOutput(dbName, abnormityType, abnormity, oModel, fModel)
        output = torch.max(output, dim=0)[0]
        output = output.unsqueeze(0)
    else:
        pass
    return output
            
        





def trainModel(device, diseaseName, dModel, oModel, fModel, criterion, optimizer, scheduler, filename, dbName, batchSize, numEpochs, gradeSize):
    criteria = getCriteria()
    dNumClasses = len(criteria) - 1
    oNumClasses = len(criteria["All"]["OCT"])
    fNumClasses = len(criteria["All"]["Fundus"])
    
    accHistory = []
    losses = []
    LRs = [optimizer.param_groups[0]["lr"]]
    startTime = time.time()
    
    oAbnormityNum = len(criteria[diseaseName]["OCT"])
    fAbnormityNum = len(criteria[diseaseName]["Fundus"])
    numClasses = oNumClasses * (oAbnormityNum == 0) + fNumClasses * (fAbnormityNum == 0)
    abnormityNum = oAbnormityNum + fAbnormityNum
    gradeLevels = abnormityNum + 1
    
    gradeTrainSize = int(0.8 * gradeSize)
    gradeValidSize = gradeSize - gradeTrainSize
    trainData = torch.empty(gradeLevels * gradeTrainSize, numClasses)
    trainLabel = torch.empty(gradeLevels * gradeValidSize, 1)
    validData = torch.empty(gradeLevels * gradeTrainSize, numClasses)
    validLabel = torch.empty(gradeLevels * gradeValidSize, 1)
    
    model = SimpleNet(numClasses, gradeLevels)
    
    for epoch in range(numEpochs):
        for grade in range(gradeLevels):
            for i in range(gradeTrainSize):
                output = getOutputAndLabel(diseaseName, oAbnormityNum, fAbnormityNum, grade, dbName)
                trainData[grade * gradeTrainSize + i] = output
                trainLabel[grade * gradeTrainSize + i] = grade
        for grade in range(gradeLevels):
            for i in range(gradeValidSize):
                output = getOutputAndLabel(diseaseName, oAbnormityNum, fAbnormityNum, grade, dbName)
                validData[grade * gradeValidSize + i] = output
                validLabel[grade * gradeValidSize + i] = grade
        
        print(f"Epoch {epoch + 1}/{numEpochs}")
        print("-" * 10)
        
        outputs = model(input_data)
        loss = criterion(outputs, target_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')
    
    timeElapsed = time.time() - startTime
    
    return model, accHistory, losses, LRs, timeElapsed
    
    startTime = time.time()
    bestAcc = 0
    model.to(device)
    validAccHistory = []
    trainAccHistory = []
    validLosses = []
    trainLosses = []
    LRs = [optimizer.param_groups[0]["lr"]]
    
    bestModelWts = copy.deepcopy(model.state_dict())
    
    dataDir = "./Data/" + dbName
    dataTransforms = transforms.Compose([transforms.Lambda(classify.resizeLongEdge), transforms.ToTensor()])
    if crossValid:
        imageDataset = datasets.ImageFolder(dataDir, dataTransforms)
    else:
        imageDatasets = {x: datasets.ImageFolder(os.path.join(dataDir, x), dataTransforms) for x in ["train", "valid"]}
        dataloaders = {x: torch.utils.data.DataLoader(imageDatasets[x], batch_size = batchSize, shuffle = True) for x in ["train", "valid"]}
    
    
    for epoch in range(numEpochs):
        print(f"Epoch {epoch + 1}/{numEpochs}")
        print("-" * 10)
        
        if crossValid:
            datasetSize = len(imageDataset)
            trainSize = int(0.8 * datasetSize)
            validSize = datasetSize - trainSize
            trainDataset, validDataset = torch.utils.data.random_split(imageDataset, [trainSize, validSize])
            dataloaders = {
                "train": torch.utils.data.DataLoader(trainDataset, batch_size = batchSize, shuffle=True),
                "valid": torch.utils.data.DataLoader(validDataset, batch_size = batchSize, shuffle=True)
            }
    
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






def train(device, featureExtract, modelName, oWts, fWts, oNumClasses, fNumClasses, dNumClasses, batchSize, gradeSize, numEpochs, LR, dbName, wtsName):
    oModel, _ = trainClassify.initializeModel(modelName, oNumClasses, featureExtract)
    oModel = oModel.to(device)
    oTrainedModel = torch.load(os.path.join(".\\TrainedModel", oWts + ".pth"))
    oModel.load_state_dict(oTrainedModel["state_dict"])
    
    fModel, _ = trainClassify.initializeModel(modelName, fNumClasses, featureExtract)
    fModel = fModel.to(device)
    fTrainedModel = torch.load(os.path.join(".\\TrainedModel", fWts + ".pth"))
    fModel.load_state_dict(fTrainedModel["state_dict"])
    
    now = datetime.now()
    filename = now.strftime("D" + " %Y-%m-%d %H-%M-%S")

    dModel = SimpleNet(oNumClasses, fNumClasses, dNumClasses)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(dModel.parameters(), lr = LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 7, gamma = 0.1)
    
    criteria = getCriteria()
    for diseaseName in criteria:
        dModel, accHistory, losses, LRs, timeElapsed = trainModel(device, diseaseName, dModel, oModel, fModel, dNumClasses, oNumClasses, fNumClasses, criterion, optimizer, scheduler, filename, dbName, batchSize, numEpochs, gradeSize)
    
    with open(os.path.join(".\\Log", filename + ".csv"), "w", newline="") as file:  
        writer = csv.writer(file)  
        writer.writerow(["Trained from scratch" if wtsName == "" else f"Trained from {wtsName}", f"batchSize = {batchSize}", f"gradeSize = {gradeSize}", f"LR = {LRs[0]}", f"epochNum = {len(numEpochs)}", f"timeElapsed = {timeElapsed // 60 :.0f}m {timeElapsed % 60: .2f}s"])
        for i in range(len(losses)):  
            writer.writerow([i + 1, accHistory[i].item(), losses[i], LRs[i]])
    print(f"Data successfully written into {filename}.csv")
    
    trainClassify.curve(accHistory, losses, filename + ".pdf")