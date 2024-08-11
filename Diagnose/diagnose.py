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
from Classify import classify
from TrainClassify import trainClassify
from TrainDiagnose import trainDiagnose
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



def diagnose(oImgs, fImgs, diseaseName, device, modelName, dWtsTime, oWts, fWts):
    criteria = trainDiagnose.getCriteria()
    oNumClasses = len(criteria["All"]["OCT"])
    fNumClasses = len(criteria["All"]["Fundus"])
    dNumClasses = len(criteria) - 1
    oAbnormityNum = len(criteria[diseaseName]["OCT"])
    fAbnormityNum = len(criteria[diseaseName]["Fundus"])
    numClasses = oAbnormityNum + fAbnormityNum
    
    if oAbnormityNum != 0:
        lenO = len(oImgs)
        oOutputs = torch.empty([lenO, oNumClasses])
        for i in range(lenO):
            img = oImgs[i]
            output = classify.classify(img, oNumClasses, device, True, modelName, oWts)
            oOutputs[i] = output
        oOutput, _ = torch.max(oOutputs, dim = 0)
    if oAbnormityNum != 0:
        lenF = len(fImgs)
        fOutputs = torch.empty([lenF, fNumClasses])
        for i in range(lenF):
            img = fImgs[i]
            output = classify.classify(img, fNumClasses, device, True, modelName, fWts)
            fOutputs[i] = output
        fOutput, _ = torch.max(fOutputs, dim = 0)
    dInput = torch.concat((oOutput, fOutput), dim = 0)
    
    dModel, _ = trainDiagnose.SimpleNet(numClasses, dNumClasses)
    dModel = dModel.to(device)
    trainedModel = torch.load(f"./TrainedModel/D {dWtsTime}/D {diseaseName} {dWtsTime}.pth")
    dModel.load_state_dict(trainedModel["state_dict"])
    
    output = dModel(dInput.to(device))
    print(output)
    return output