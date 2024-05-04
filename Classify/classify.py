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

def processImg(imgPath):
    img = Image.open(imgPath)
    img = train.resizeLongEdge(imgPath)
    toTensor = transforms.ToTensor()
    img = toTensor(img)
    return img

def imgShow(img, ax = None, title = None):
    if ax is None:
        fig, ax = plt.subplots()
    img = np.array(img).transpose((1, 2, 0))
    ax.imshow(img)
    ax.set_title(title)
    return ax

def classify(imgName, numClasses, device, useGpu, featureExtract, modelName, filename):
    imgPath = os.path.join("./NewImage", imgName)
    modelFt, inputSize = train.initializeModel(modelName, numClasses, featureExtract)
    modelFt = modelFt.to(device)
    trainedModel = torch.load(filename + ".pth")
    bestAcc = trainedModel["best_acc"]
    modelFt.load_state_dict(trainedModel["state_dict"])
    img = processImg(imgPath)
    imgShow(img)
    output = modelFt(img.cuda() if useGpu else img)
    print(output.shape)
    