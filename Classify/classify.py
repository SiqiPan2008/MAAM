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
from Train import train
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def processImg(imgPath, customResize = 224):
    img = Image.open(imgPath)
    if customResize != 0:
        img = train.resizeLongEdge(img, longEdgeSize = customResize)
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

def classify(imgPath, numClasses, device, useGpu, featureExtract, modelName, wtsName):
    modelFt, _ = train.initializeModel(modelName, numClasses, featureExtract)
    modelFt = modelFt.to(device)
    trainedModel = torch.load(os.path.join(".\\TrainedModel", wtsName + ".pth"))
    modelFt.load_state_dict(trainedModel["state_dict"])
    img = processImg(imgPath)
    imgShow(img)
    img = img.unsqueeze(0)
    output = modelFt(img.cuda() if useGpu else img)
    print(output)