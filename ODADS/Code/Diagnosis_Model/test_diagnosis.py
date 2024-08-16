import os
import torch
import time
import numpy as np
from ODADS.Code.Utils import utils
from ODADS.Code.Diagnosis_Model import diagnosis_model, train_diagnosis
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
def testAbnormityNumModel(device, diseaseName, oFoldername, oName, oClassSize, fFoldername, fName, fClassSize, dWtsDTime, batchSize, gradeSize):
    criteria = utils.get_criteria()
    oAbnormityNum = len(criteria[diseaseName]["OCT"])
    fAbnormityNum = len(criteria[diseaseName]["Fundus"])
    allAbnormities = [("Fundus", abnormity) for abnormity in criteria["All"]["Fundus"]] + \
                     [("OCT", abnormity) for abnormity in criteria["All"]["OCT"]]
    allOAbnormityNum = len(criteria["All"]["OCT"])
    allFAbnormityNum = len(criteria["All"]["Fundus"])
    allAbnormityNum = len(allAbnormities)
    gradeLevels = oAbnormityNum + fAbnormityNum + 1
    
    outputPathO = os.path.join("ODADS/Data/Results", os.path.join(oFoldername, oName + ".bin"))
    outputsO = np.fromfile(outputPathO, dtype = np.float64).reshape((allOAbnormityNum, oClassSize, allOAbnormityNum))
    outputPathF = os.path.join("ODADS/Data/Results", os.path.join(fFoldername, fName + ".bin"))
    outputsF = np.fromfile(outputPathF, dtype = np.float64).reshape((allFAbnormityNum, fClassSize, allFAbnormityNum))
    
    testData = torch.zeros(gradeLevels * gradeSize, allAbnormityNum)
    testLabel = torch.zeros(gradeLevels * gradeSize, dtype=torch.long)
    
    dModel = diagnosis_model.Simple_Net(allAbnormityNum, gradeLevels)
    trainedModel = torch.load(os.path.join("ODADS/Data/Weights", os.path.join(dWtsDTime, f"{dWtsDTime} {diseaseName}.pth")))
    dModel.load_state_dict(trainedModel["state_dict"])
    
    startTime = time.time()
    
    for grade in range(gradeLevels):
        for i in range(gradeSize):
            output = train_diagnosis.getOutputsFromFile(allAbnormities, diseaseName, oAbnormityNum, fAbnormityNum, grade, outputsO, outputsF)
            testData[grade * gradeSize + i] = output
            testLabel[grade * gradeSize + i] = grade
    testDataset = torch.utils.data.TensorDataset(testData, testLabel)
    testLoader = torch.utils.data.DataLoader(testDataset, batch_size = batchSize, shuffle = True)
    
    dModel.eval()
    corrects = 0
    total = len(testLoader.dataset)
    with torch.no_grad():
        for inputs, labels in testLoader:
            outputs = dModel(inputs)
            _, predicted = torch.max(outputs, 1)
            corrects += (predicted == labels).sum().item()
    acc = corrects / total
    timeElapsed = time.time() - startTime
    print(f"Time elapsed {timeElapsed // 60 :.0f}m {timeElapsed % 60 :.2f}s")
    print(f"Acc: {corrects}/{total} = {acc :.4f}")
    
    return acc

        
def testDiseaseProbModel(device, oFoldername, oName, oClassSize, fFoldername, fName, fClassSize, dWtsDTime, ddWtsDTime, batchSize, diseaseSize):
    criteria = utils.get_criteria()
    diseaseIncludingNormal = [disease for disease in criteria.keys() if disease != "All"]
    diseaseNum = len(diseaseIncludingNormal)
    abnormityVectorSize = sum([(len(criteria[disease]["OCT"]) + len(criteria[disease]["Fundus"]) + 1) \
        for disease in criteria.keys() if disease not in ["Normal", "All"]])
    testData = torch.zeros(diseaseNum * diseaseSize, abnormityVectorSize)
    testLabel = torch.zeros(diseaseNum * diseaseSize, dtype=torch.long)
    
    allAbnormities = [("Fundus", abnormity) for abnormity in criteria["All"]["Fundus"]] + \
                    [("OCT", abnormity) for abnormity in criteria["All"]["OCT"]]
    allOAbnormityNum = len(criteria["All"]["OCT"])
    allFAbnormityNum = len(criteria["All"]["Fundus"])
    allAbnormityNum = len(allAbnormities)
    outputPathO = os.path.join("ODADS/Data/Results", os.path.join(oFoldername, oName + ".bin"))
    outputsO = np.fromfile(outputPathO, dtype = np.float64).reshape((allOAbnormityNum, oClassSize, allOAbnormityNum))
    outputPathF = os.path.join("ODADS/Data/Results", os.path.join(fFoldername, fName + ".bin"))
    outputsF = np.fromfile(outputPathF, dtype = np.float64).reshape((allFAbnormityNum, fClassSize, allFAbnormityNum))
    
    ddModel = diagnosis_model.SimpleNet(abnormityVectorSize, diseaseNum).to(device)
    trainedModel = torch.load(os.path.join("ODADS/Data/Weights", os.path.join(ddWtsDTime, f"{ddWtsDTime}.pth")))
    ddModel.load_state_dict(trainedModel["state_dict"])
    
    dModels = {disease: diagnosis_model.SimpleNet(allAbnormityNum, len(criteria[disease]["OCT"]) + len(criteria[disease]["Fundus"]) + 1).to(device) \
         for disease in criteria.keys() if disease not in ["Normal", "All"]}
    for disease in dModels.keys():
         trainedModel = torch.load(os.path.join("ODADS/Data/Weights", os.path.join(dWtsDTime, f"{dWtsDTime} {disease}.pth")))
         dModels[disease].load_state_dict(trainedModel["state_dict"])
    
    startTime = time.time()
    for i in range(len(diseaseIncludingNormal)):
        disease = diseaseIncludingNormal[i]
        for j in range(diseaseSize):
            output = train_diagnosis.getAbnormityNumsVectorFromFile(device, disease, outputsO, outputsF, allAbnormities, dModels)
            output = output.detach()
            testData[i * diseaseSize + j] = output
            testLabel[i * diseaseSize + j] = i
    testDataset = torch.utils.data.TensorDataset(testData, testLabel)
    testLoader = torch.utils.data.DataLoader(testDataset, batch_size = batchSize, shuffle = True)

    ddModel.eval()
    corrects = 0
    total = len(testLoader.dataset)
    with torch.no_grad():
        for inputs, labels in testLoader:
            outputs = ddModel(inputs.to(device))
            _, predicted = torch.max(outputs, 1)
            corrects += (predicted.cpu() == labels).sum().item()
    acc = corrects / total
    timeElapsed = time.time() - startTime
    print(f"Time elapsed {timeElapsed // 60 :.0f}m {timeElapsed % 60 :.2f}s")
    print(f"acc: {corrects}/{total} = {acc :.4f}")
    
    return acc






def test(device, diseaseName, oFoldername, oName, oClassSize, fFoldername, fName, fClassSize, batchSize, classSize, dWtsDTime, ddWtsDTime = None):
    if diseaseName == "all disease prob":
        acc = testDiseaseProbModel(device, oFoldername, oName, oClassSize, fFoldername, fName, fClassSize, dWtsDTime, ddWtsDTime, batchSize, classSize)
    else:
        acc = testAbnormityNumModel(device, diseaseName, oFoldername, oName, oClassSize, fFoldername, fName, fClassSize, dWtsDTime, batchSize, classSize)
    
    return acc