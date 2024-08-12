import os
import torch
from torch import nn
import torch.optim as optim
from torchvision import transforms, datasets
import time
import copy
import csv
from datetime import datetime
from Utils import utils
from AbnormityModels import abnormityModel
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



def trainModel(device, model, criterion, optimizer, scheduler, filename, dataDir, crossValid, batchSize, numEpochs, isInception = False):
    startTime = time.time()
    bestAcc = 0
    model.to(device)
    validAccHistory = []
    trainAccHistory = []
    validLosses = []
    trainLosses = []
    LRs = [optimizer.param_groups[0]["lr"]]
    
    bestModelWts = copy.deepcopy(model.state_dict())
    
    dataTransforms = transforms.Compose([transforms.Lambda(utils.resizeLongEdge), transforms.ToTensor()])
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
                torch.save(state, f"ODADS/Data/Weights/{filename}/{filename}.pth")
                print(f"Data successfully written into {filename}.pth")
                
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
    print(f"best valid acc: {bestAcc :.4f}")
    model.load_state_dict(bestModelWts)
    
    return model, validAccHistory, trainAccHistory, validLosses, trainLosses, LRs, timeElapsed

def train(device, featureExtract, modelName, numClasses, batchSize, numEpochs, LR, usePretrained, dbName, wtsName, modelType, crossValid = True):
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
    
    now = datetime.now()
    filename = now.strftime(modelType + " %Y-%m-%d %H-%M-%S")

    model, validAccHistory, trainAccHistory, validLosses, trainLosses, LRs, timeElapsed = trainModel(device, model, criterion, optimizer, scheduler, filename + ".pth", dbName, crossValid, batchSize, numEpochs)
    with open(f"ODADS/Data/Results/{filename}/{filename}.csv", "w", newline="") as file:  
        writer = csv.writer(file)  
        writer.writerow(["Trained from ResNet152" if wtsName == "" else f"Trained from {wtsName}", f"Data: {dbName}", f"batchSize = {batchSize}", f"LR = {LRs[0]}", f"epochNum = {len(trainLosses)}", f"timeElapsed = {timeElapsed // 60 :.0f}m {timeElapsed % 60: .2f}s"])
        for i in range(len(trainLosses)):  
            writer.writerow([i + 1, validAccHistory[i].item(), trainAccHistory[i].item(), validLosses[i], trainLosses[i], LRs[i]])
    print(f"Data successfully written into {filename}.csv")
    
    utils.curve(validAccHistory, trainAccHistory, validLosses, trainLosses, f"ODADS/Data/Results/{filename}/{filename}.pdf")