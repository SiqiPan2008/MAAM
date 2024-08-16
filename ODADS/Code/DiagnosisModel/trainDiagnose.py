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
    
    
    

def trainAbnormityNumModel(device, diseaseName, oFoldername, oName, oClassSize, fFoldername, fName, fClassSize, dWtsDTime, dTime, batchSize, LR, numEpochs, gradeSize):
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
    if dWtsDTime:
        trainedModel = torch.load(os.path.join("ODADS/Data/Weights", os.path.join(dWtsDTime, f"{diseaseName} {dWtsDTime}.pth")))
        dModel.load_state_dict(trainedModel["state_dict"])
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






def getAbnormityNumsVectorFromFile(device, diseaseName, outputsO, outputsF, allAbnormities, dModels):
    criteria = utils.getCriteria()
    oAbnormityNum = len(criteria[diseaseName]["OCT"])
    fAbnormityNum = len(criteria[diseaseName]["Fundus"])
    abnormityOutput = getOutputsFromFile(allAbnormities, diseaseName, oAbnormityNum, fAbnormityNum, oAbnormityNum + fAbnormityNum, outputsO, outputsF)
    abnormityOutput = abnormityOutput.unsqueeze(0).type(torch.float32).to(device)
    abnormityNumsVector = torch.empty([0]).to(device)
    for disease in dModels.keys():
        tmpModel = dModels[disease]
        tmpModel.eval()
        abnormityNumOutput = tmpModel(abnormityOutput)[0]
        abnormityNumOutput = torch.nn.functional.softmax(abnormityNumOutput, 0)
        abnormityNumsVector = torch.cat([abnormityNumsVector, abnormityNumOutput])
    return abnormityNumsVector
        
def trainDiseaseProbModel(device, oFoldername, oName, oClassSize, fFoldername, fName, fClassSize, dWtsDTime, ddWtsDTime, batchSize, LR, numEpochs, diseaseSize):
    criteria = utils.getCriteria()
    diseaseIncludingNormal = [disease for disease in criteria.keys() if disease != "All"]
    diseaseNum = len(diseaseIncludingNormal)
    abnormityVectorSize = sum([(len(criteria[disease]["OCT"]) + len(criteria[disease]["Fundus"]) + 1) \
        for disease in criteria.keys() if disease not in ["Normal", "All"]])
    diseaseTrainSize = int(0.8 * diseaseSize)
    diseaseValidSize = diseaseSize - diseaseTrainSize
    trainData = torch.zeros(diseaseNum * diseaseTrainSize, abnormityVectorSize)
    trainLabel = torch.zeros(diseaseNum * diseaseTrainSize, dtype=torch.long)
    validData = torch.zeros(diseaseNum * diseaseValidSize, abnormityVectorSize)
    validLabel = torch.zeros(diseaseNum * diseaseValidSize, dtype=torch.long)
    
    allAbnormities = [("Fundus", abnormity) for abnormity in criteria["All"]["Fundus"]] + \
                    [("OCT", abnormity) for abnormity in criteria["All"]["OCT"]]
    allOAbnormityNum = len(criteria["All"]["OCT"])
    allFAbnormityNum = len(criteria["All"]["Fundus"])
    allAbnormityNum = len(allAbnormities)
    outputPathO = os.path.join("ODADS/Data/Results", os.path.join(oFoldername, oName + ".bin"))
    outputsO = np.fromfile(outputPathO, dtype = np.float64).reshape((allOAbnormityNum, oClassSize, allOAbnormityNum))
    outputPathF = os.path.join("ODADS/Data/Results", os.path.join(fFoldername, fName + ".bin"))
    outputsF = np.fromfile(outputPathF, dtype = np.float64).reshape((allFAbnormityNum, fClassSize, allFAbnormityNum))
    
    ddModel = diagnosisModel.SimpleNet(abnormityVectorSize, diseaseNum).to(device)
    if ddWtsDTime:
        trainedModel = torch.load(os.path.join("ODADS/Data/Weights", os.path.join(ddWtsDTime, f"D{ddWtsDTime}.pth")))
        ddModel.load_state_dict(trainedModel["state_dict"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddModel.parameters(), lr = LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 7, gamma = 0.1)
    
    dModels = {disease: diagnosisModel.SimpleNet(allAbnormityNum, len(criteria[disease]["OCT"]) + len(criteria[disease]["Fundus"]) + 1).to(device) \
         for disease in criteria.keys() if disease not in ["Normal", "All"]}
    for disease in dModels.keys():
         trainedModel = torch.load(os.path.join("ODADS/Data/Weights", os.path.join(dWtsDTime, f"{dWtsDTime} {disease}.pth")))
         dModels[disease].load_state_dict(trainedModel["state_dict"])
    
    startTime = time.time()
    bestAcc = 0.0
    bestModelWts = copy.deepcopy(ddModel.state_dict())
    LRs = [optimizer.param_groups[0]["lr"]]
    
    trainAccHistory = []
    validAccHistory = []
    trainLosses = []
    validLosses = []
    for epoch in range(numEpochs):
        for i in range(diseaseNum):
            disease = diseaseIncludingNormal[i]
            for j in range(diseaseTrainSize):
                output = getAbnormityNumsVectorFromFile(device, disease, outputsO, outputsF, allAbnormities, dModels)
                output = output.detach()
                trainData[i * diseaseTrainSize + j] = output
                trainLabel[i * diseaseTrainSize + j] = i
        for i in range(len(diseaseIncludingNormal)):
            disease = diseaseIncludingNormal[i]
            for j in range(diseaseValidSize):
                output = getAbnormityNumsVectorFromFile(device, disease, outputsO, outputsF, allAbnormities, dModels)
                output = output.detach()
                validData[i * diseaseValidSize + j] = output
                validLabel[i * diseaseValidSize + j] = i
        trainDataset = torch.utils.data.TensorDataset(trainData, trainLabel)
        validDataset = torch.utils.data.TensorDataset(validData, validLabel)
        trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size = batchSize, shuffle = True)
        validLoader = torch.utils.data.DataLoader(validDataset, batch_size = batchSize, shuffle = True)
        
        print(f"Epoch {epoch + 1}/{numEpochs}")
        print("-" * 10)

        ddModel.train()
        runningLoss = 0.0
        corrects = 0
        total = len(trainLoader.dataset)
        for inputs, labels in trainLoader:
            optimizer.zero_grad()
            outputs = ddModel(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            runningLoss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            corrects += (predicted.cpu() == labels).sum().item()
        acc = corrects / total
        trainAccHistory.append(acc)
        runningLoss = runningLoss / total
        trainLosses.append(runningLoss)
        timeElapsed = time.time() - startTime
        print(f"Time elapsed {timeElapsed // 60 :.0f}m {timeElapsed % 60 :.2f}s")
        print(f"Train loss: {runningLoss :.4f}, acc: {acc :.4f}")
        
        ddModel.eval()
        corrects = 0
        total = len(validLoader.dataset)
        valRunningLoss = 0.0
        with torch.no_grad():
            for inputs, labels in validLoader:
                outputs = ddModel(inputs.to(device))
                loss = criterion(outputs, labels.to(device))
                valRunningLoss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                corrects += (predicted.cpu() == labels).sum().item()
        acc = corrects / total
        validAccHistory.append(acc)
        valRunningLoss = valRunningLoss / total
        validLosses.append(valRunningLoss)
        timeElapsed = time.time() - startTime
        print(f"Time elapsed {timeElapsed // 60 :.0f}m {timeElapsed % 60 :.2f}s")
        print(f"Valid loss: {valRunningLoss :.4f}, acc: {acc :.4f}")
        
        if acc >= bestAcc:
            bestAcc = acc
            bestModelWts = copy.deepcopy(ddModel.state_dict())
            state = {
                "state_dict": ddModel.state_dict(),
                "best_acc": bestAcc,
                "optimizer": optimizer.state_dict()
            }
            os.makedirs(f"ODADS/Data/Weights/D{dWtsDTime}/", exist_ok=True)
            torch.save(state, os.path.join(f"ODADS/Data/Weights/D{dWtsDTime}/D{dWtsDTime}.pth"))
            print(f"Data successfully written into ODADS/Data/Weights/D{dWtsDTime}/D{dWtsDTime}.pth")
        
        scheduler.step()
        LRs.append(optimizer.param_groups[0]["lr"])
        print(f"optimizer learning rate: {LRs[-1]:.7f}")
        print()
    
    timeElapsed = time.time() - startTime
    print(f"training complete in {timeElapsed // 60 :.0f}m {timeElapsed % 60 :.2f}s")
    print(f"best valid acc: {bestAcc :.4f}")
    ddModel.load_state_dict(bestModelWts)
    
    return ddModel, trainAccHistory, validAccHistory, trainLosses, validLosses, LRs, timeElapsed






def train(device, diseaseName, oFoldername, oName, oClassSize, fFoldername, fName, fClassSize, batchSize, classSize, numEpochs, LR, dWtsDTime, dTime, ddWtsDTime = None):
    filename = f"D{dTime}" if diseaseName == "all disease prob" else f"D {dTime} {diseaseName}"
    
    if diseaseName == "all disease prob":
        dModel, trainAccHistory, validAccHistory, trainLosses, validLosses, LRs, timeElapsed = trainDiseaseProbModel(device, oFoldername, oName, oClassSize, fFoldername, fName, fClassSize, dWtsDTime, ddWtsDTime, batchSize, LR, numEpochs, classSize)
    else:
        dModel, trainAccHistory, validAccHistory, trainLosses, validLosses, LRs, timeElapsed = trainAbnormityNumModel(device, diseaseName, oFoldername, oName, oClassSize, fFoldername, fName, fClassSize, dWtsDTime, dTime, batchSize, LR, numEpochs, classSize)

    os.makedirs(f"ODADS/Data/Results/D {dTime}/", exist_ok=True)
    with open(f"ODADS/Data/Results/D {dTime}/{filename}.csv", "w", newline="") as file:  
        writer = csv.writer(file)  
        writer.writerow([
            "Trained from scratch" if dWtsDTime == "" else f"Trained from {dWtsDTime}", 
            f"oFolder: {oFoldername}", 
            f"oName : {oName}", 
            f"fFolder: {fFoldername}", 
            f"fName : {fName}", 
            f"batchSize = {batchSize}", 
            f"LR = {LRs[0]}", 
            f"epochNum = {len(trainLosses)}", 
            f"classSize = {classSize}", 
            f"timeElapsed = {timeElapsed // 60 :.0f}m {timeElapsed % 60: .2f}s"
        ])
        for i in range(len(trainLosses)):  
            writer.writerow([i + 1, validAccHistory[i], trainAccHistory[i], validLosses[i], trainLosses[i], LRs[i]])
    print(f"Data successfully written into {filename}.csv")
    print("\n" for _ in range(3))
    
    utils.curve(validAccHistory, trainAccHistory, validLosses, trainLosses, f"ODADS/Data/Results/D {dTime}/{filename}.pdf")