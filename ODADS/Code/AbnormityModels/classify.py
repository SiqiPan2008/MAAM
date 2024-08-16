import os
import torch
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
from AbnormityModels import abnormityModel
from Utils import utils
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def classify(img, model, device):
    model.eval()
    img = utils.resizeLongEdge(img)
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    output = model(img.to(device))
    output = torch.nn.functional.softmax(output[0], dim=0)
    # print(output)
    return output

def classifyImg(img, numClasses, device, featureExtract, modelName, foldername, wtsName):
    model, _ = abnormityModel.initializeAbnormityModel(modelName, numClasses, featureExtract)
    model = model.to(device)
    trainedModel = torch.load(f"ODADS/Data/Weights/{foldername}/{wtsName}")
    model.load_state_dict(trainedModel["state_dict"])
    # utils.imgShow(img)
    return classify(img, model, device)

def classifyDatabase(dbName, numClasses, device, featureExtract, modelName, foldername, wtsName):
    model, _ = abnormityModel.initializeAbnormityModel(modelName, numClasses, featureExtract)
    model = model.to(device)
    trainedModel = torch.load(f"ODADS/Data/Weights/{foldername}{wtsName}")
    model.load_state_dict(trainedModel["state_dict"])
    classNames = [f for f in os.scandir(dbName) if f.is_dir()]
    total = 0
    corrects = 0
    
    outputs = []
    for label in range(len(classNames)):
        classFolder = classNames[label].path
        imgPaths = [f for f in os.listdir(classFolder) if any(f.endswith(imgSuffix) for imgSuffix in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"])]
        classOutput = []
        for i in range(len(imgPaths)):
            model.eval()
            img = Image.open(os.path.join(classFolder, imgPaths[i])).convert("RGB")
            img = utils.resizeLongEdge(img)
            img = transforms.ToTensor()(img)
            img = img.unsqueeze(dim = 0)
            img = img.to(device)
            output = model(img)
            output = torch.nn.functional.softmax(output[0], dim = 0)
            classOutput.append(output.cpu().detach().tolist())
            _, pred = torch.max(output, dim=0)
            corrects += pred == label
            total += 1
            if ((i + 1) % 1000 == 0):
                print(i + 1)
        outputs.append(classOutput)
        print(classNames[label].name)
        
    np.array(outputs).tofile(f"ODADS/Data/Results/{foldername}/{wtsName}.bin")
    print(f"corrects: {corrects}/{total}")
    print(f"accuracy: {corrects/total}")


def testAccWithLoader(device, featureExtract, modelName, numClasses, dbPath, dbName, foldername, wtsName): # only top prob
    model, _ = abnormityModel.initializeAbnormityModel(modelName, numClasses, featureExtract)
    model = model.to(device)
    trainedModel = torch.load(f"ODADS/Data/Weights/{foldername}/{wtsName}")
    model.load_state_dict(trainedModel["state_dict"])
    model.eval()
    transform = transforms.Compose([
                    transforms.Lambda(lambda x: utils.resizeLongEdge(x)),
                    transforms.ToTensor()
                ])
    imageDataset = datasets.ImageFolder(dbPath, transform)
    dataloader = torch.utils.data.DataLoader(imageDataset, batch_size = 1, shuffle=True)
    
    outputs = [[] for _ in range(numClasses)]
    runningCorrects = 0
    total = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            batchOutputs = model(inputs)
            output = torch.nn.functional.softmax(batchOutputs[0], dim = 0)
            outputs[labels[0]].append(output.cpu().detach().tolist())
            preds = torch.max(batchOutputs, 1)[1]
        total += len(preds)
        runningCorrects += torch.sum(preds == labels.data)
        if total % 500 == 0:
            print(f"{runningCorrects}/{total}")
            print(f"{runningCorrects/total}")
    accuracy = runningCorrects / total
    np.array(outputs).tofile(f"ODADS/Data/Results/{foldername}/{dbName} {wtsName[:-4]}.bin")
    print(f"corrects: {runningCorrects}/{total}")
    print(f"accuracy: {accuracy}")
    return runningCorrects, total, accuracy