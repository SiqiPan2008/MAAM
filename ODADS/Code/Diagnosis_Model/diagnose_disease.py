import os
import torch
from ODADS.Code.Abnormity_Models import classify_abnormity
from ODADS.Code.Diagnosis_Model import diagnosis_model
from ODADS.Code.Utils import utils
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# small utilities. refactor later!
# OUTDATED! rewrite

def diagnose(oImgs, fImgs, diseaseName, device, modelName, dWtsTime, oWts, fWts):
    criteria = utils.get_criteria()
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
            output = classify_abnormity.get_abnormity_probs_from_img(img, oNumClasses, device, True, modelName, oWts)
            oOutputs[i] = output
        oOutput, _ = torch.max(oOutputs, dim = 0)
    if oAbnormityNum != 0:
        lenF = len(fImgs)
        fOutputs = torch.empty([lenF, fNumClasses])
        for i in range(lenF):
            img = fImgs[i]
            output = classify_abnormity.get_abnormity_probs_from_img(img, fNumClasses, device, True, modelName, fWts)
            fOutputs[i] = output
        fOutput, _ = torch.max(fOutputs, dim = 0)
    dInput = torch.concat((oOutput, fOutput), dim = 0)
    
    dModel, _ = diagnosis_model.Simple_Net(abnormityNum, gradeLevels)
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