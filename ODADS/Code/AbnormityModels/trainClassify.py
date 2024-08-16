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

def train(name, device):
    setting = utils.getSetting()
    LR = setting.LR
    batch_size = setting.batch_size
    cross_valid = setting.use_cross_valid
    feature_extract = setting.feature_extract
    save_model_frequency = setting.save_model_frequency
    net_name = setting.get_net(name)
    num_epochs = setting.get_num_epochs(name)
    img_folder = setting.get_img_folder(name)
    folder_path = setting.get_folder_path(name)
    wt_file_name = setting.get_wt_file_name(name)
    rs_file_name = setting.get_rs_file_name(name)
    num_classes = setting.get_num_abnormities(name)
    use_pretrained = setting.is_transfer_learning(name)
    is_transfer_learning = setting.is_transfer_learning(name)
    
    model, _ = abnormityModel.initializeAbnormityModel(net_name, num_classes, feature_extract, use_pretrained)
    model = model.to(device)
    if not is_transfer_learning:
        trainedModel = torch.load(os.path.join(folder_path, setting.get_transfer_learning_wt(name) + ".pth"))
        model.load_state_dict(trainedModel['state_dict']) 
    paramsToUpdate = model.parameters()
    print("Params to learn:")
    if feature_extract:
        paramsToUpdate = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if feature_extract:
                paramsToUpdate.append(param)
            print("\t", name)
            
    optimizer = optim.Adam(paramsToUpdate, lr = LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 7, gamma = 0.1)
    criterion = nn.CrossEntropyLoss()
    
    startTime = time.time()
    lastTime = startTime
    bestAcc = [0]
    model.to(device)
    validAccHistory = []
    trainAccHistory = []
    validLosses = []
    trainLosses = []
    LRs = [optimizer.param_groups[0]["lr"]]
    
    dataTransforms = transforms.Compose([transforms.ToTensor()])
    if cross_valid:
        imageDataset = datasets.ImageFolder(img_folder, dataTransforms)
    else:
        imageDatasets = {x: datasets.ImageFolder(os.path.join(img_folder, x), dataTransforms) for x in ["train", "valid"]}
        dataloaders = {x: torch.utils.data.DataLoader(imageDatasets[x], batch_size = batch_size, shuffle = True) for x in ["train", "valid"]}
    
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)
        
        if cross_valid:
            labels = np.array(imageDataset.targets)
            trainIndices = []
            validIndices = []
            for classId in range(num_classes):
                classIndices = np.where(labels == classId)[0]
                np.random.shuffle(classIndices)
                splitPoint = int(0.8 * len(classIndices))
                trainIndices.extend(classIndices[:splitPoint])
                validIndices.extend(classIndices[splitPoint:])
            trainDataset = Subset(imageDataset, trainIndices)
            validDataset = Subset(imageDataset, validIndices)
            dataloaders = {
                "train": torch.utils.data.DataLoader(trainDataset, batch_size = batch_size, shuffle=True),
                "valid": torch.utils.data.DataLoader(validDataset, batch_size = batch_size, shuffle=True)
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
            totalTimeElapsed = time.time() - startTime
            timeElapsed = time.time() - lastTime
            lastTime = time.time()
            print(f"total time elapsed {totalTimeElapsed // 60 :.0f}m {totalTimeElapsed % 60 :.2f}s")
            print(f"time elapsed {timeElapsed // 60 :.0f}m {totalTimeElapsed % 60 :.2f}s")
            print(f"{phase} loss: {epochLoss :.4f}, acc: {epochAcc :.4f}")
            
            if phase == "valid" and epochAcc >= bestAcc[-1]:
                bestAcc[-1] = epochAcc
                state = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict()
                }
                epochRange = int(epoch / save_model_frequency) * save_model_frequency
                torch.save(state, os.path.join(folder_path, wt_file_name + f" {epochRange: 3d}.pth"))
                print(f"Data successfully written into {wt_file_name}.pth")
            if phase == "valid" and (epoch + 1) % save_model_frequency == 0:
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
        
    totalTimeElapsed = time.time() - startTime
    print(f"training complete in {totalTimeElapsed // 60 :.0f}m {totalTimeElapsed % 60 :.2f}s")
    print(f"best valid acc: {bestAcc}")

    with open(os.path.join(folder_path, rs_file_name + ".csv"), "w", newline="") as file:  
        writer = csv.writer(file)  
        for i in range(len(trainLosses)):  
            writer.writerow([i + 1, validAccHistory[i].item(), trainAccHistory[i].item(), validLosses[i], trainLosses[i], LRs[i]])
    print(f"Data successfully written into {rs_file_name}.csv")
    
    utils.curve(validAccHistory, trainAccHistory, validLosses, trainLosses, os.path.join(folder_path, rs_file_name + ".pdf"))