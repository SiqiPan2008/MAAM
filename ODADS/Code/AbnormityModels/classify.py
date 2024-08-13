import os
import torch
from AbnormityModels import abnormityModel
from Utils import utils
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def classify(img, numClasses, device, featureExtract, modelName, wtsName):
    model, _ = abnormityModel.initializeAbnormityModel(modelName, numClasses, featureExtract)
    model = model.to(device)
    trainedModel = torch.load(f"ODADS/Data/Weights/{wtsName}/{wtsName}.pth")
    model.load_state_dict(trainedModel["state_dict"])
    # utils.imgShow(img)
    img = utils.resizeLongEdge(img)
    img = img.unsqueeze(0)
    output = model(img.to(device))
    print(output[0])
    return(output[0])