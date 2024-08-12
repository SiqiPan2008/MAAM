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
import initAbnormityModel
from Utils import utils
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def classify(img, numClasses, device, featureExtract, modelName, wtsName):
    modelFt, _ = initAbnormityModel.initializeAbnormityModel(modelName, numClasses, featureExtract)
    modelFt = modelFt.to(device)
    trainedModel = torch.load(os.path.join(".\\TrainedModel", wtsName + ".pth"))
    modelFt.load_state_dict(trainedModel["state_dict"])
    #utils.imgShow(img)
    img = img.unsqueeze(0)
    output = modelFt(img.to(device))
    print(output[0])
    return(output[0])