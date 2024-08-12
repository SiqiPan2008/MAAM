import os
import torch
from AbnormityModels import abnormityModel
from Utils import utils
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def classify(img, numClasses, device, featureExtract, modelName, wtsName):
    modelFt, _ = abnormityModel.initializeAbnormityModel(modelName, numClasses, featureExtract)
    modelFt = modelFt.to(device)
    trainedModel = torch.load(os.path.join(".\\TrainedModel", wtsName + ".pth"))
    modelFt.load_state_dict(trainedModel["state_dict"])
    #utils.imgShow(img)
    img = utils.resizeLongEdge(img) # Necessary?
    img = img.unsqueeze(0)
    output = modelFt(img.to(device))
    print(output[0])
    return(output[0])