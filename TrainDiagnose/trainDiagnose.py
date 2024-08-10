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

def getRandImageOutput(device, dbName, abnormity, oModel, fModel):
    abnormityType, abnormityName = abnormity[0], abnormity[1]
    foldername = f"{dbName}/{abnormityType}/{abnormityName}"
    files = os.listdir(foldername)
    randomImg = random.choice(files)
    imgPath = os.path.join(foldername, randomImg)
    img = classify.processImg(imgPath)
    img = img.unsqueeze(0)
    output = oModel(img.to(device)) if abnormityType == "OCT" else fModel(img.to(device))
    return output

def getOutputAndLabel(device, diseaseName, oAbnormityNum, fAbnormityNum, grade, dbName, oModel, fModel):
    criteria = getCriteria()
   
    diseaseAbnormities = [(("Fundus", abnormity) for abnormity in criteria[diseaseName]["Fundus"])
                            + (("OCT", abnormity) for abnormity in criteria[diseaseName]["OCT"])]
    selCorrectAbnormities = random.sample(diseaseAbnormities, grade)
    correctFs = sum((abnormity[0] == "Fundus") for abnormity in diseaseAbnormities)
    correctOs = sum((abnormity[0] == "OCT") for abnormity in diseaseAbnormities)
    
    allFAbnormity = [("Fundus", abnormity) for abnormity in criteria["All"]["Fundus"]]
    allOAbnormity = [("OCT", abnormity) for abnormity in criteria["All"]["OCT"]]
    incorrectFAbnormities = [abnormity for abnormity in allFAbnormity if (abnormity not in diseaseAbnormities)]
    incorrectOAbnormities = [abnormity for abnormity in allOAbnormity if (abnormity not in diseaseAbnormities)]
    selIncorrectFAbnormities = random.sample(incorrectFAbnormities, fAbnormityNum - correctFs)
    selIncorrectOAbnormities = random.sample(incorrectOAbnormities, oAbnormityNum - correctOs)
    
    selAbnormities = selCorrectAbnormities + selIncorrectFAbnormities + selIncorrectOAbnormities
    
    outputs = torch.empty([fAbnormityNum, len(allFAbnormity) + len(allOAbnormity)])
    for i in range(len(selAbnormities)):
        outputs[i] = getRandImageOutput(device, dbName, selAbnormities[i], oModel, fModel)
    outputs = torch.max(outputs, dim=0)
    outputs = outputs.unsqueeze(0)
    return outputs
            
        





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
    trainLabel = torch.empty(gradeLevels * gradeValidSize)
    validData = torch.empty(gradeLevels * gradeTrainSize, numClasses)
    validLabel = torch.empty(gradeLevels * gradeValidSize)
    
    model = SimpleNet(numClasses, gradeLevels)
    
    trainAcc = []
    validAcc = []
    trainLosses = []
    validLosses = []
    for epoch in range(numEpochs):
        for grade in range(gradeLevels):
            for i in range(gradeTrainSize):
                output = getOutputAndLabel(device, diseaseName, oAbnormityNum, fAbnormityNum, grade, dbName, oModel, fModel)
                trainData[grade * gradeTrainSize + i] = output
                trainLabel[grade * gradeTrainSize + i] = grade
        for grade in range(gradeLevels):
            for i in range(gradeValidSize):
                output = getOutputAndLabel(device, diseaseName, oAbnormityNum, fAbnormityNum, grade, dbName, oModel, fModel)
                validData[grade * gradeValidSize + i] = output
                validLabel[grade * gradeValidSize + i] = grade
        trainDataset = torch.utils.data.TensorDataset(trainData, validData)
        validDataset = torch.utils.data.TensorDataset(trainData, validData)
        trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size = batchSize, shuffle = False)
        validLoader = torch.utils.data.DataLoader(validDataset, batch_size = batchSize, shuffle = False)
        
        print(f"Epoch {epoch + 1}/{numEpochs}")
        print("-" * 10)

        model.train()  # 设为训练模式
        runningLoss = 0.0
        corrects = 0
        total = 0
        for inputs, labels in trainLoader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            runningLoss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            corrects += (predicted == labels).sum().item()
        acc = corrects / total
        trainAcc.append(acc)
        
        loss = runningLoss / len(trainLoader.dataset)
        trainLosses.append(loss)
        timeElapsed = time.time() - startTime
        print(f"time elapsed {timeElapsed // 60 :.0f}m {timeElapsed % 60 :.2f}s")
        print(f"train loss: {loss :.4f}, acc: {acc :.4f}")
        
        model.eval()
        corrects = 0
        total = 0
        valRunningLoss = 0.0
        with torch.no_grad():
            for inputs, labels in validLoader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valRunningLoss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                corrects += (predicted == labels).sum().item()
        loss = valRunningLoss / len(validLoader.dataset)
        acc = corrects / total
        validAcc.append(acc)
        validLosses.append(loss)
        timeElapsed = time.time() - startTime
        print(f"time elapsed {timeElapsed // 60 :.0f}m {timeElapsed % 60 :.2f}s")
        print(f"train loss: {loss :.4f}, acc: {acc :.4f}")
        
    # record LR
    # save model
    # record best acc
    
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
    optimizer = optim.Adam(dModel.parameters(), lr = LR)
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