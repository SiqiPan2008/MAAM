import os
import torch
from torch import nn
import torch.optim as optim
import time
import random
import copy
from PIL import Image
import csv
from datetime import datetime
from AbnormityModels import abnormityModel
from Utils import utils
from DiagnosisModel import diagnosisModel
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def getOutputs(device, diseaseName, oAbnormityNum, fAbnormityNum, grade, dbName, oModel, fModel):
    criteria = utils.getCriteria()
   
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
        abnormityType, abnormityName = selAbnormities[i][0], selAbnormities[i][1]
        foldername = f"{dbName}/{abnormityType}/{abnormityName}"
        files = os.listdir(foldername)
        randomImg = random.choice(files)
        imgPath = os.path.join(foldername, randomImg)
        img = Image.open(imgPath)
        outputs[i] = utils.getRandImageOutput(device, dbName, img, abnormityType, oModel, fModel)
    outputs = torch.max(outputs, dim=0)
    outputs = outputs.unsqueeze(0)
    return outputs
            
def trainModel(device, diseaseName, oModel, fModel, wtsName, dTime, dbName, batchSize, LR, numEpochs, gradeSize):
    criteria = utils.getCriteria()
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
    
    dModel = diagnosisModel.SimpleNet(abnormityNum, gradeLevels)
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
                output = getOutputs(device, diseaseName, oAbnormityNum, fAbnormityNum, grade, dbName, oModel, fModel)
                trainData[grade * gradeTrainSize + i] = output
                trainLabel[grade * gradeTrainSize + i] = grade
        for grade in range(gradeLevels):
            for i in range(gradeValidSize):
                output = getOutputs(device, diseaseName, oAbnormityNum, fAbnormityNum, grade, dbName, oModel, fModel)
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
            os.makedirs(f"ODADS/Data/Weights/D {dTime}/", exist_ok=True)
            torch.save(state, os.path.join(f"ODADS/Data/Weights/D {dTime}/D {diseaseName} {dTime}.pth"))
            print(f"Data successfully written into ODADS/Data/Weights/D {dTime}/D {diseaseName} {dTime}.pth")
        
        scheduler.step()
        LRs.append(optimizer.param_groups[0]["lr"])
        print(f"optimizer learning rate: {LRs[-1]:.7f}")
        print()
    
    timeElapsed = time.time() - startTime
    print(f"training complete in {timeElapsed // 60 :.0f}m {timeElapsed % 60 :.2f}s")
    print(f"best valid acc: {bestAcc :.4f}")
    dModel.load_state_dict(bestModelWts)
    
    return dModel, trainAccHistory, validAccHistory, trainLosses, validLosses, LRs, timeElapsed

def train(device, diseaseName, featureExtract, modelName, oWts, fWts, batchSize, gradeSize, numEpochs, LR, dbName, wtsName, dTime):
    criteria = utils.getCriteria()
    oNumClasses = len(criteria["All"]["OCT"])
    fNumClasses = len(criteria["All"]["Fundus"])
    
    oModel, _ = abnormityModel.initializeAbnormityModel(modelName, oNumClasses, featureExtract)
    oModel = oModel.to(device)
    oTrainedModel = torch.load(f"ODADS/Data/Weights/{oWts}/{oWts}.pth")
    oModel.load_state_dict(oTrainedModel["state_dict"])
    
    fModel, _ = abnormityModel.initializeAbnormityModel(modelName, fNumClasses, featureExtract)
    fModel = fModel.to(device)
    fTrainedModel = torch.load(f"ODADS/Data/Weights/{fWts}/{fWts}.pth")
    fModel.load_state_dict(fTrainedModel["state_dict"])
    
    filename = f"D {diseaseName} {dTime}"
    
    dModel, validAccHistory, trainAccHistory, trainLosses, validLosses, LRs, timeElapsed = trainModel(device, diseaseName, oModel, fModel, wtsName, dTime, dbName, batchSize, LR, numEpochs, gradeSize)
    
    os.makedirs(f"ODADS/Data/Results/D {dTime}/", exist_ok=True)
    with open(f"ODADS/Data/Results/D {dTime}/{filename}.csv", "w", newline="") as file:  
        writer = csv.writer(file)  
        writer.writerow(["Trained from scratch" if wtsName == "" else f"Trained from {wtsName}", f"Data: {dbName}", f"batchSize = {batchSize}", f"LR = {LRs[0]}", f"epochNum = {len(trainLosses)}", f"gradeSize = {gradeSize}", f"timeElapsed = {timeElapsed // 60 :.0f}m {timeElapsed % 60: .2f}s"])
        for i in range(len(trainLosses)):  
            writer.writerow([i + 1, validAccHistory[i].item(), trainAccHistory[i].item(), validLosses[i], trainLosses[i], LRs[i]])
    print(f"Data successfully written into {filename}.csv")
    
    utils.curve(validAccHistory, trainAccHistory, validLosses, trainLosses, f"ODADS/Data/Results/D {dTime}/{filename}.pdf")