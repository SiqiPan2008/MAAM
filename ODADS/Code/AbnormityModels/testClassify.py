import os
import torch
from torchvision import transforms, datasets
from PIL import Image
from Utils import utils
from AbnormityModels import abnormityModel, classify
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def testAcc(device, featureExtract, modelName, numClasses, dbName, wtsName):
    model, _ = abnormityModel.initializeAbnormityModel(modelName, numClasses, featureExtract)
    model = model.to(device)
    trainedModel = torch.load(f"ODADS/Data/Weights/{wtsName}/{wtsName}.pth")
    model.load_state_dict(trainedModel["state_dict"])
    transform = transforms.Compose([
                    transforms.Lambda(lambda x: utils.resizeLongEdge(x)),
                    transforms.ToTensor()
                ])
    imageDataset = datasets.ImageFolder(dbName, transform)
    classToIdx = imageDataset.class_to_idx
    
    allClasses = [f.name for f in os.scandir(dbName) if f.is_dir()]
    corrects = 0
    total = 0
    for className in allClasses:
        classFolder = f"{dbName}{className}/"
        for imgName in os.listdir(classFolder):
            img = Image.open(f"{classFolder}{imgName}/")
            output = classify.classify(img, model, device)
            _, pred = torch.max(output, dim=0)
            corrects += pred == classToIdx[className]
            total += 1
    accuracy = corrects / total
    print(f"corrects: {corrects}/{total}")
    print(f"accuracy: {accuracy}")
    return accuracy