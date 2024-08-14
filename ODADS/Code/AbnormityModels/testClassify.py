import os
import torch
import csv
from torchvision import transforms, datasets
from PIL import Image
from Utils import utils
from AbnormityModels import abnormityModel, classify
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def testAcc(device, featureExtract, modelName, numClasses, dbName, foldername, wtsName):
    model, _ = abnormityModel.initializeAbnormityModel(modelName, numClasses, featureExtract)
    model = model.to(device)
    trainedModel = torch.load(f"ODADS/Data/Weights/{foldername}/{wtsName}")
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
        print(f"completed {className}")
    accuracy = corrects / total
    print(f"corrects: {corrects}/{total}")
    print(f"accuracy: {accuracy}")
    return corrects, total, accuracy

def testMultipleAcc(device, featureExtract, modelName, numClasses, dbName, foldername):
    wtsFolder = f"ODADS/Data/Weights/{foldername}" 
    with open(f"ODADS/Data/Results/{foldername}/{foldername} Test Acc.csv", "w", newline="") as file:  
        writer = csv.writer(file)
        writer.writerow(["Filename", "Corrects", "Total", "Accuracy"])
        for filename in os.listdir(wtsFolder):
            if os.path.splitext(filename)[1] == ".pth":
                corrects, total, accuracy = testAcc(device, featureExtract, modelName, numClasses, dbName, foldername, filename)
            writer.writerow([filename, corrects, total, accuracy])
    print(f"Data successfully written into {filename}.csv")