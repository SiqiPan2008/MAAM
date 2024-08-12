from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def getCriteria():
    with open('Criteria.json', 'r', encoding='utf-8') as file:
        criteria = json.load(file)
    return criteria

def resizeLongEdge(img, longEdgeSize = 224):
    width, height = img.size
    if width > height:
        newSize = (longEdgeSize, int(height * longEdgeSize / width))
        loc = (0, int((longEdgeSize - newSize[1]) / 2))
    else:
        newSize = (int(longEdgeSize * width / height), longEdgeSize)
        width, _ = img.size
        loc = (int((longEdgeSize - newSize[0]) / 2), 0)
    img = img.resize(newSize)
    blackBackground = Image.new("RGB", (longEdgeSize, longEdgeSize), "black")
    blackBackground.paste(img, loc)
    return blackBackground

def curve(validAccHistory, trainAccHistory, validLosses, trainLosses, filename, show = False):
    plt.clf()
    x = range(1, len(trainLosses) + 1)
    
    vAH = [validAccHistory[i].item() for i in range(len(x))]
    tAH = [trainAccHistory[i].item() for i in range(len(x))]
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
    plt.savefig(filename, format="pdf")
    if show:
        plt.show()

def processImg(img, customResize = 224):
    if customResize != 0:
        img = resizeLongEdge(img, longEdgeSize = customResize)
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
    
def getRandImageOutput(device, dbName, img, abnormityType, oModel, fModel):
    img = processImg(img)
    img = img.unsqueeze(0)
    output = oModel(img.to(device)) if abnormityType == "OCT" else fModel(img.to(device))
    return output