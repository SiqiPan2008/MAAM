import os
import torch
from torchvision import transforms
from AbnormityModels import abnormityModel
from Utils import utils
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def classify(img, model, device):
    img = utils.resizeLongEdge(img)
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    output = model(img.to(device))
    output = torch.exp(output[0])
    # print(output)
    return output

def classifyImg(img, numClasses, device, featureExtract, modelName, wtsName):
    model, _ = abnormityModel.initializeAbnormityModel(modelName, numClasses, featureExtract)
    model = model.to(device)
    trainedModel = torch.load(f"ODADS/Data/Weights/{wtsName}/{wtsName}.pth")
    model.load_state_dict(trainedModel["state_dict"])
    # utils.imgShow(img)
    return classify(img, model, device)