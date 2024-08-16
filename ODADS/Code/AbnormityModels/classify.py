import os
import torch
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
from AbnormityModels import abnormityModel
from Utils import utils
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# small utilities. refactor later!

def get_abnormities_probs(img, model, device):
    model.eval()
    img = utils.resize_long_edge(img)
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    output = model(img.to(device))
    output = torch.nn.functional.softmax(output[0], dim=0)
    print(output)
    return output

def get_abnormity_probs_from_img(img, numClasses, device, featureExtract, modelName, foldername, wtsName):
    model, _ = abnormityModel.initializeAbnormityModel(modelName, numClasses, featureExtract)
    model = model.to(device)
    trainedModel = torch.load(f"ODADS/Data/Weights/{foldername}/{wtsName}")
    model.load_state_dict(trainedModel["state_dict"])
    utils.img_show(img)
    return get_abnormities_probs(img, model, device)