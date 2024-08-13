import os
from torch import nn
from torchvision import models
from Utils import utils
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def setParameterDoNotRequireGrad(model, featureExtract):
    if featureExtract:
        for param in model.parameters():
            param.requires_grad = False

def initializeAbnormityModel(modelName, numClasses, featureExtract, usePretrained = True):
    model = None
    inputSize = 0
    
    if modelName == "resnet":
        model = models.resnet152(weights = models.ResNet152_Weights.DEFAULT if usePretrained else None)
        setParameterDoNotRequireGrad(model, featureExtract if usePretrained else False)
        numFtrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(numFtrs, numClasses), nn.LogSoftmax(dim = 1))
        inputSize = 224
    else:
        print("Invalid model name.")
        exit()
    
    return model, inputSize