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
    img = classify.processImgFromPath(imgPath)
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
    
    oOutputs = torch.empty([fAbnormityNum, len(allFAbnormity) + len(allOAbnormity)])
    for i in range(len(selAbnormities)):
        outputs[i] = getRandImageOutput(device, dbName, selAbnormities[i], oModel, fModel)
    outputs = torch.max(outputs, dim=0)
    outputs = outputs.unsqueeze(0)
    return outputs
            
def trainModel(device, diseaseName, oModel, fModel, wtsName, filename, dbName, batchSize, LR, numEpochs, gradeSize):
    criteria = getCriteria()
    dNumClasses = len(criteria) - 1
    oNumClasses = len(criteria["All"]["OCT"])
    fNumClasses = len(criteria["All"]["Fundus"])
    
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
    
    dModel = SimpleNet(abnormityNum, dNumClasses)
    if wtsName:
        dModel.load_state_dict("wtsName")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(dModel.parameters(), lr = LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 7, gamma = 0.1)
    
    startTime = time.time()
    bestAcc = 0.0
    bestModelWts = copy.deepcopy(dModel.state_dict())
    LRs = [optimizer.param_groups[0]["lr"]]
    accHistory = []
    losses = []
    
    trainAccHistory = []
    validAccHistory = []
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

        dModel.train()
        runningLoss = 0.0
        corrects = 0
        total = len(trainLoader.dataset)
        for inputs, labels in trainLoader:
            optimizer.zero_grad()
            outputs = dModel(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            runningLoss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            corrects += (predicted == labels).sum().item()
        acc = corrects / total
        trainAccHistory.append(acc)
        loss = runningLoss / total
        trainLosses.append(loss)
        timeElapsed = time.time() - startTime
        print(f"Time elapsed {timeElapsed // 60 :.0f}m {timeElapsed % 60 :.2f}s")
        print(f"Train loss: {loss :.4f}, acc: {acc :.4f}")
        
        dModel.eval()
        corrects = 0
        total = len(validLoader.dataset)
        valRunningLoss = 0.0
        with torch.no_grad():
            for inputs, labels in validLoader:
                outputs = dModel(inputs)
                loss = criterion(outputs, labels)
                valRunningLoss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                corrects += (predicted == labels).sum().item()
        acc = corrects / total
        validAccHistory.append(acc)
        loss = valRunningLoss / total
        validLosses.append(loss)
        timeElapsed = time.time() - startTime
        print(f"Time elapsed {timeElapsed // 60 :.0f}m {timeElapsed % 60 :.2f}s")
        print(f"Valid loss: {loss :.4f}, acc: {acc :.4f}")
        
        if acc >= bestAcc:
            bestAcc = acc
            bestModelWts = copy.deepcopy(dModel.state_dict())
            state = {
                "state_dict": dModel.state_dict(),
                "best_acc": bestAcc,
                "optimizer": optimizer.state_dict()
            }
            torch.save(state, os.path.join(".\\TrainedModel", filename))
            print(f"Data successfully written into {filename}")
        
        scheduler.step()
        LRs.append(optimizer.param_groups[0]["lr"])
        print(f"optimizer learning rate: {LRs[-1]:.7f}")
        print()
    
    timeElapsed = time.time() - startTime
    print(f"training complete in {timeElapsed // 60 :.0f}m {timeElapsed % 60 :.2f}s")
    print(f"best valid acc: {bestAcc :.4f}")
    dModel.load_state_dict(bestModelWts)
    
    return dModel, trainAccHistory, validAccHistory, trainLosses, validLosses, LRs, timeElapsed

def train(device, diseaseName, featureExtract, modelName, oWts, fWts, batchSize, gradeSize, numEpochs, LR, dbName, wtsName):
    criteria = getCriteria()
    oNumClasses = len(criteria["All"]["OCT"])
    fNumClasses = len(criteria["All"]["Fundus"])
    
    oModel, _ = trainClassify.initializeModel(modelName, oNumClasses, featureExtract)
    oModel = oModel.to(device)
    oTrainedModel = torch.load(os.path.join(".\\TrainedModel", oWts + ".pth"))
    oModel.load_state_dict(oTrainedModel["state_dict"])
    
    fModel, _ = trainClassify.initializeModel(modelName, fNumClasses, featureExtract)
    fModel = fModel.to(device)
    fTrainedModel = torch.load(os.path.join(".\\TrainedModel", fWts + ".pth"))
    fModel.load_state_dict(fTrainedModel["state_dict"])
    
    now = datetime.now()
    filename = now.strftime("D " + diseaseName + " %Y-%m-%d %H-%M-%S")
    
    dModel, validAccHistory, trainAccHistory, trainLosses, validLosses, LRs, timeElapsed = trainModel(device, diseaseName, oModel, fModel, wtsName, "./TrainedModel/D %Y-%m-%d %H-%M-%S/" + filename + ".pth", dbName, batchSize, LR, numEpochs, gradeSize)
    
    with open(os.path.join(".\\Log", filename + ".csv"), "w", newline="") as file:  
        writer = csv.writer(file)  
        writer.writerow(["Trained from scratch" if wtsName == "" else f"Trained from {wtsName}", f"Data: {dbName}", f"batchSize = {batchSize}", f"LR = {LRs[0]}", f"epochNum = {len(trainLosses)}", f"gradeSize = {gradeSize}", f"timeElapsed = {timeElapsed // 60 :.0f}m {timeElapsed % 60: .2f}s"])
        for i in range(len(trainLosses)):  
            writer.writerow([i + 1, validAccHistory[i].item(), trainAccHistory[i].item(), validLosses[i], trainLosses[i], LRs[i]])
    print(f"Data successfully written into {filename}.csv")
    
    trainClassify.curve(validAccHistory, trainAccHistory, validLosses, trainLosses, filename + ".pdf")