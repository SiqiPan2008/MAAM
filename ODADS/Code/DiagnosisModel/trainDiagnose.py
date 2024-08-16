import os
import torch
from torch import nn
import torch.optim as optim
import time
import random
import copy
from PIL import Image
import csv
import numpy as np
from AbnormityModels import abnormityModel
from Utils import utils
from DiagnosisModel import diagnosisModel
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def getOutputs(device, diseaseName, oAbnormityNum, fAbnormityNum, grade, dbName, oModel, fModel):
    criteria = utils.getCriteria() 
    correctAbnormities = [("Fundus", abnormity) for abnormity in criteria[diseaseName]["Fundus"]] + \
                         [("OCT", abnormity) for abnormity in criteria[diseaseName]["OCT"]]
    allAbnormities = [("Fundus", abnormity) for abnormity in criteria["All"]["Fundus"]] + \
                     [("OCT", abnormity) for abnormity in criteria["All"]["OCT"]]
    incorrectAbnormities = [abnormity for abnormity in allAbnormities if abnormity not in correctAbnormities]
    selCorrectAbnormities = random.sample(correctAbnormities, grade)
    selIncorrectAbnormities = random.sample(incorrectAbnormities, oAbnormityNum + fAbnormityNum - grade)
    selAbnormities = selCorrectAbnormities + selIncorrectAbnormities
    
    oOutput = torch.zeros([len(criteria["All"]["OCT"])]).to(device)
    fOutput = torch.zeros([len(criteria["All"]["Fundus"])]).to(device)
    for abnormity in selAbnormities:
        abnormityType, abnormityName = abnormity[0], abnormity[1]
        foldername = f"{dbName}/{abnormityType}/{abnormityName}"
        files = os.listdir(foldername)
        randomImg = random.choice(files)
        imgPath = os.path.join(foldername, randomImg)
        img = Image.open(imgPath)
        output = utils.getRandImageOutput(device, dbName, img, abnormityType, oModel, fModel)
        if abnormityType == "OCT":
            oOutput = torch.maximum(oOutput, output)
        elif abnormityType == "Fundus":
            fOutput = torch.maximum(fOutput, output)
    output = torch.concat([fOutput, oOutput])
    return output

def getOutputsFromFile(allAbnormities, diseaseName, oAbnormityNum, fAbnormityNum, grade, outputsO, outputsF):
    criteria = utils.getCriteria() 
    correctAbnormities = [("Fundus", abnormity) for abnormity in criteria[diseaseName]["Fundus"]] + \
                         [("OCT", abnormity) for abnormity in criteria[diseaseName]["OCT"]]
    incorrectAbnormities = [abnormity for abnormity in allAbnormities if abnormity not in correctAbnormities]
    selCorrectAbnormities = random.sample(correctAbnormities, grade)
    selIncorrectAbnormities = random.sample(incorrectAbnormities, oAbnormityNum + fAbnormityNum - grade)
    selAbnormities = selCorrectAbnormities + selIncorrectAbnormities
    
    oOutput = torch.zeros([len(criteria["All"]["OCT"])])
    fOutput = torch.zeros([len(criteria["All"]["Fundus"])])
    for abnormity in selAbnormities:
        abnormityType, abnormityName = abnormity[0], abnormity[1]
        if abnormityType == "OCT":
            output = random.choice(outputsO[criteria["All"]["OCT"].index(abnormityName)])
            oOutput = torch.max(torch.tensor(output), oOutput)
        elif abnormityType == "Fundus":
            output = random.choice(outputsF[criteria["All"]["Fundus"].index(abnormityName)])
            fOutput = torch.max(torch.tensor(output), fOutput)
    output = torch.concat([fOutput, oOutput])
    return output
    
    
    

def trainModel(device, diseaseName, oFoldername, oName, oClassSize, fFoldername, fName, fClassSize, dWtsName, dTime, batchSize, LR, numEpochs, gradeSize):
    criteria = utils.getCriteria()
    oAbnormityNum = len(criteria[diseaseName]["OCT"])
    fAbnormityNum = len(criteria[diseaseName]["Fundus"])
    allAbnormities = [("Fundus", abnormity) for abnormity in criteria["All"]["Fundus"]] + \
                     [("OCT", abnormity) for abnormity in criteria["All"]["OCT"]]
    allOAbnormityNum = len(criteria["All"]["OCT"])
    allFAbnormityNum = len(criteria["All"]["Fundus"])
    allAbnormityNum = len(allAbnormities)
    gradeLevels = oAbnormityNum + fAbnormityNum + 1
    
    outputPathO = os.path.join("ODADS/Data/Results", os.path.join(oFoldername, oName + ".bin"))
    outputsO = np.fromfile(outputPathO, dtype = np.float64).reshape((allOAbnormityNum, oClassSize, allOAbnormityNum))
    outputPathF = os.path.join("ODADS/Data/Results", os.path.join(fFoldername, fName + ".bin"))
    outputsF = np.fromfile(outputPathF, dtype = np.float64).reshape((allFAbnormityNum, fClassSize, allFAbnormityNum))
    
    gradeTrainSize = int(0.8 * gradeSize)
    gradeValidSize = gradeSize - gradeTrainSize
    trainData = torch.zeros(gradeLevels * gradeTrainSize, allAbnormityNum)
    trainLabel = torch.zeros(gradeLevels * gradeTrainSize, dtype=torch.long)
    validData = torch.zeros(gradeLevels * gradeValidSize, allAbnormityNum)
    validLabel = torch.zeros(gradeLevels * gradeValidSize, dtype=torch.long)
    
    dModel = diagnosisModel.SimpleNet(allAbnormityNum, gradeLevels)
    if dWtsName:
        dModel.load_state_dict(os.path.join("ODADS/Data/Weights", os.path.join(dWtsName, f"{diseaseName} {dWtsName}.pth")))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(dModel.parameters(), lr = LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 7, gamma = 0.1)
    
    startTime = time.time()
    bestAcc = 0.0
    bestModelWts = copy.deepcopy(dModel.state_dict())
    LRs = [optimizer.param_groups[0]["lr"]]
    
    trainAccHistory = []
    validAccHistory = []
    trainLosses = []
    validLosses = []
    for epoch in range(numEpochs):
        for grade in range(gradeLevels):
            for i in range(gradeTrainSize):
                output = getOutputsFromFile(allAbnormities, diseaseName, oAbnormityNum, fAbnormityNum, grade, outputsO, outputsF)
                trainData[grade * gradeTrainSize + i] = output
                trainLabel[grade * gradeTrainSize + i] = grade
        for grade in range(gradeLevels):
            for i in range(gradeValidSize):
                output = getOutputsFromFile(allAbnormities, diseaseName, oAbnormityNum, fAbnormityNum, grade, outputsO, outputsF)
                validData[grade * gradeValidSize + i] = output
                validLabel[grade * gradeValidSize + i] = grade
        trainDataset = torch.utils.data.TensorDataset(trainData, trainLabel)
        validDataset = torch.utils.data.TensorDataset(validData, validLabel)
        trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size = batchSize, shuffle = True)
        validLoader = torch.utils.data.DataLoader(validDataset, batch_size = batchSize, shuffle = True)
        
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
            torch.save(state, os.path.join(f"ODADS/Data/Weights/D {dTime}/D {dTime} {diseaseName}.pth"))
            print(f"Data successfully written into ODADS/Data/Weights/D {dTime}/D {dTime} {diseaseName}.pth")
        
        scheduler.step()
        LRs.append(optimizer.param_groups[0]["lr"])
        print(f"optimizer learning rate: {LRs[-1]:.7f}")
        print()
    
    timeElapsed = time.time() - startTime
    print(f"training complete in {timeElapsed // 60 :.0f}m {timeElapsed % 60 :.2f}s")
    print(f"best valid acc: {bestAcc :.4f}")
    dModel.load_state_dict(bestModelWts)
    
    return dModel, trainAccHistory, validAccHistory, trainLosses, validLosses, LRs, timeElapsed

def train(device, diseaseName, oFoldername, oName, oClassSize, fFoldername, fName, fClassSize, dBatchSize, gradeSize, numEpochs, LR, dWtsName, dTime):
    filename = f"D {dTime} {diseaseName}"
    
    dModel, trainAccHistory, validAccHistory, trainLosses, validLosses, LRs, timeElapsed = trainModel(device, diseaseName, oFoldername, oName, oClassSize, fFoldername, fName, fClassSize, dWtsName, dTime, dBatchSize, LR, numEpochs, gradeSize)
    
    os.makedirs(f"ODADS/Data/Results/D {dTime}/", exist_ok=True)
    with open(f"ODADS/Data/Results/D {dTime}/{filename}.csv", "w", newline="") as file:  
        writer = csv.writer(file)  
        writer.writerow([
            "Trained from scratch" if dWtsName == "" else f"Trained from {dWtsName}", 
            f"oFolder: {oFoldername}", 
            f"oName : {oName}", 
            f"fFolder: {fFoldername}", 
            f"fName : {fName}", 
            f"batchSize = {dBatchSize}", 
            f"LR = {LRs[0]}", 
            f"epochNum = {len(trainLosses)}", 
            f"gradeSize = {gradeSize}", 
            f"timeElapsed = {timeElapsed // 60 :.0f}m {timeElapsed % 60: .2f}s"
        ])
        for i in range(len(trainLosses)):  
            writer.writerow([i + 1, validAccHistory[i], trainAccHistory[i], validLosses[i], trainLosses[i], LRs[i]])
    print(f"Data successfully written into {filename}.csv")
    
    utils.curve(validAccHistory, trainAccHistory, validLosses, trainLosses, f"ODADS/Data/Results/D {dTime}/{filename}.pdf")