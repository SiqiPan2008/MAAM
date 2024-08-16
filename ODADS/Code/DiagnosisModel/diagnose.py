import os
import torch
from AbnormityModels import classify
from DiagnosisModel import trainDiagnose
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

### OUTDATED!!!!!

def diagnose(oImgs, fImgs, diseaseName, device, modelName, dWtsTime, oWts, fWts):
    criteria = trainDiagnose.utils.getCriteria()
    oNumClasses = len(criteria["All"]["OCT"])
    fNumClasses = len(criteria["All"]["Fundus"])
    oAbnormityNum = len(criteria[diseaseName]["OCT"])
    fAbnormityNum = len(criteria[diseaseName]["Fundus"])
    abnormityNum = oAbnormityNum + fAbnormityNum
    gradeLevels = abnormityNum + 1
    
    if oAbnormityNum != 0:
        lenO = len(oImgs)
        oOutputs = torch.empty([lenO, oNumClasses])
        for i in range(lenO):
            img = oImgs[i]
            output = classify.classifyImg(img, oNumClasses, device, True, modelName, oWts)
            oOutputs[i] = output
        oOutput, _ = torch.max(oOutputs, dim = 0)
    if oAbnormityNum != 0:
        lenF = len(fImgs)
        fOutputs = torch.empty([lenF, fNumClasses])
        for i in range(lenF):
            img = fImgs[i]
            output = classify.classifyImg(img, fNumClasses, device, True, modelName, fWts)
            fOutputs[i] = output
        fOutput, _ = torch.max(fOutputs, dim = 0)
    dInput = torch.concat((oOutput, fOutput), dim = 0)
    
    dModel, _ = trainDiagnose.diagnosisModel.SimpleNet(abnormityNum, gradeLevels)
    dModel = dModel.to(device)
    trainedModel = torch.load(f"ODADS/Data/Weights/D {dWtsTime}/D {diseaseName} {dWtsTime}.pth")
    dModel.load_state_dict(trainedModel["state_dict"])
    
    output = dModel(dInput.to(device))[0]
    print(output)
    expectedVal = 0
    for grade in range(len(output)):
        expectedVal += grade * output[grade]
    print(expectedVal)
    return output, expectedVal