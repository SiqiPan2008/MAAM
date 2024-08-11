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
from TrainClassify import trainClassify
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def processImg(img, customResize = 224):
    if customResize != 0:
        img = trainClassify.resizeLongEdge(img, longEdgeSize = customResize)
    toTensor = transforms.ToTensor()
    img = toTensor(img)
    return img

def imgShow(img, ax = None, title = None):
    if ax is None:
        fig, ax = plt.subplots()
    img = np.array(img).transpose((1, 2, 0))
    ax.imshow(img)
    ax.set_title(title)
    plt.show()

def classify(img, numClasses, device, featureExtract, modelName, wtsName):
    modelFt, _ = trainClassify.initializeModel(modelName, numClasses, featureExtract)
    modelFt = modelFt.to(device)
    trainedModel = torch.load(os.path.join(".\\TrainedModel", wtsName + ".pth"))
    modelFt.load_state_dict(trainedModel["state_dict"])
    #imgShow(img)
    img = img.unsqueeze(0)
    output = modelFt(img.to(device))
    print(output[0])
    return(output[0])