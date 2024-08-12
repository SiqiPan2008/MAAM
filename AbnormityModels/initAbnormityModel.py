import os
from torch import nn
from torchvision import models
from Utils import utils
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def setParameterRequiresGrad(model, featureExtract):
    if featureExtract:
        for param in model.parameters():
            param.requires_grad = False

def initializeAbnormityModel(modelName, numClasses, featureExtract, usePretrained = True):
    modelFt = None
    inputSize = 0
    
    if modelName == "resnet":
        modelFt = models.resnet152(weights = models.ResNet152_Weights.DEFAULT if usePretrained else None)
        utils.setParameterRequiresGrad(modelFt, featureExtract if usePretrained else False)
        numFtrs = modelFt.fc.in_features
        modelFt.fc = nn.Sequential(nn.Linear(numFtrs, numClasses), nn.LogSoftmax(dim = 1))
        inputSize = 224
    else:
        print("Invalid model name.")
        exit()
    
    return modelFt, inputSize