from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import json
import setting
from typing import Dict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# read Criteria.json and get criteria
def getCriteria():
    with open(r'ODADS\Code\Criteria.json', 'r', encoding='utf-8') as file:
        criteria = json.load(file)
    return criteria

# load json file
def load_json(filename: str) -> Dict:
    with open(filename, 'r') as file:
        return json.load(file)

# read Setting.json
def getSetting():
    json_data = load_json('ODADS/CODE/Setting.json')
    return setting.Setting.from_dict(json_data)

# resize the image so that its longer edge has length "longEdgeSize"
# then paste it on a square black background
def resizeLongEdge(img, longEdgeSize = 224):
    width, height = img.size
    if width > height:
        newSize = (longEdgeSize, int(height * longEdgeSize / width))
        loc = (0, int((longEdgeSize - newSize[1]) / 2))
    else:
        newSize = (int(longEdgeSize * width / height), longEdgeSize)
        loc = (int((longEdgeSize - newSize[0]) / 2), 0)
    img = img.resize(newSize)
    blackBackground = Image.new("RGB", (longEdgeSize, longEdgeSize), "black")
    blackBackground.paste(img, loc)
    return blackBackground

# use valid and train accuracies and losses to draw curves
def curve(validAccHistory, trainAccHistory, validLosses, trainLosses, rs_file_name, show = False):
    plt.clf()
    x = range(1, len(trainLosses) + 1)
    
    vAH = [validAccHistory[i] for i in range(len(x))]
    tAH = [trainAccHistory[i] for i in range(len(x))]
    plt.subplot(2, 1, 1)
    plt.plot(x, vAH, label = "validAcc")
    plt.plot(x, tAH, label = "trainAcc")
    plt.title("Accuracy Curve")  
    plt.xlabel("Epoch")  
    plt.ylabel("Accuracy")  
    plt.legend()  
      
    plt.subplot(2, 1, 2)
    plt.plot(x, validLosses, label = "validLoss")
    plt.plot(x, trainLosses, label = "trainLoss")
    plt.title("Loss Curve")  
    plt.xlabel("Epoch")  
    plt.ylabel("Loss")  
    plt.legend() 
    
    plt.tight_layout()
    plt.savefig(rs_file_name, format="pdf")
    if show:
        plt.show()

# resizeLongEdge() an image and then convert it to Tensor
def resizeAndToTensor(img, customResize = 224):
    if customResize != 0:
        img = resizeLongEdge(img, longEdgeSize = customResize)
    toTensor = transforms.ToTensor()
    img = toTensor(img)
    return img

# show an image in Matplotlib
def imgShow(img, ax = None, title = None):
    if ax is None:
        fig, ax = plt.subplots()
    img = np.array(img).transpose((1, 2, 0))
    ax.imshow(img)
    ax.set_title(title)
    plt.show()

# get the least number of probabilities so that their sum is greater tham minScore
# returns the indices of the top probabilities
def getTopProbIndices(output, allowNum, minScore):
    sortedOutput, indices = torch.sort(output, descending=True)
    sum = 0
    topIndices = []
    for i in range(allowNum):
        if sum >= minScore:
            break
        sum += sortedOutput[i]
        topIndices.append(indices[i])
    return topIndices
        
