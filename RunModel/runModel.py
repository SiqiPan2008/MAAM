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
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def setParameterRequiresGrad(model, featureExtract):
    if featureExtract:
        for param in model.parameters():
            param.requires_grad = False

def initializeModel (modelName, numClasses, featureExtract, usePretrained = True):
    modelFt = None
    inputSize = 0
    
    if modelName == "resnet":
        modelFt = models.resnet152(pretrained = usePretrained)
        setParameterRequiresGrad(modelFt, featureExtract)
        numFtrs = modelFt.fc.in_features
        modelFt.fc = nn.Sequential(nn.Linear(numFtrs, numClasses), nn.LogSoftmax(dim = 1))
        inputSize = 224
    else:
        print("Invalid model name.")
        exit()
    
    return modelFt, inputSize

def trainModel(device, model, dataloaders, criterion, optimizer, scheduler, filename, numEpochs = 25, isInception = False):
    startTime = time.time()
    bestAcc = 0
    model.to(device)
    testAccHistory = []
    trainAccHistory = []
    testLosses = []
    trainLosses = []
    LRs = [optimizer.param_groups[0]["lr"]]
    
    bestModelWts = copy.deepcopy(model.state_dict())
    
    for epoch in range(numEpochs):
        print(f"Epoch {epoch + 1}/{numEpochs}")
        print("-" * 10)
        
        for phase in ["train", "test"]:
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
                        loss.backward
                        optimizer.step()
                runningLoss += loss.item() * inputs.size(0) # what does 0 mean
                runningCorrects += torch.sum(preds = labels.data)
            
            datasetLen = len(dataloaders[phase].dataset)
            epochLoss = runningLoss / datasetLen
            epochAcc = runningCorrects / datasetLen
            timeElapsed = time.time() - startTime
            print(f"time elapsed {timeElapsed // 60 :.0f}m {timeElapsed % 60 :.2f}")
            print(f"{phase} loss: {epochLoss :.4f}, acc: {epochAcc :.4f}")
            
                        
        
    

def runModel():
    dataDir = "../data"
    trainDir = dataDir + "/train"
    testDir = dataDir + "/test"
    batchSize = 16
    dataTransforms = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor()])
    imageDatasets = {x: datasets.ImageFolder(os.path.join(dataDir, x), dataTransforms) for x in ["train", "test"]}
    dataloaders = {x: torch.utils.data.DataLoader(imageDatasets[x], batchSize = batchSize, shuffle = True) for x in ["train", "test"]}
    datasetSizes = {x: len(imageDatasets[x]) for x in ["train", "test"]}
    classNames = imageDatasets["train"].classes
    
    modelName = "resnet"
    featureExtract = True
    useGpu = torch.cuda.is_available()
    print("CUDA available." if useGpu else "CUDA not available.")
    device = torch.device("cuda:0" if useGpu else "cpu") # get to know how to use both GPUs
    print(device)
    
    modelFt, inputSize = initializeModel(modelName, 128, featureExtract) # what does FT stand for?
    modelFt = modelFt.to(device)
    filename = "checkpoint.pth"
    
    paramsToUpdate = modelFt.parameters()
    print("Params to learn:")
    if featureExtract:
        paramsToUpdate = []
    for name, param in modelFt.named_parameters():
        if param.requires_grad:
            if featureExtract:
                paramsToUpdate.append(param)
            print("\t", name)
            
    optimizerFt = optim.Adam(paramsToUpdate, lr = 1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizerFt, step_size = 7, gamma = 0.1)
    criterion = nn.NLLLoss() # what is NLL?
    
    return