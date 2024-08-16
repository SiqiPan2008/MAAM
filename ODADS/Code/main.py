from AbnormityModels import classify, trainClassify, testClassify, gradcam
from DiagnosisModel import diagnose, trainDiagnose
from Utils import utils
import torch
import numpy as np
from PIL import Image
from datetime import datetime
import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    modelName = "resnet"
    task = sys.argv[1]
    cudaDevice = sys.argv[2]
    featureExtract = True # True -> freeze conv layers; False -> train all layers
    useGpu = torch.cuda.is_available()
    print("CUDA available." if useGpu else
          "CUDA not available.")
    device = torch.device(cudaDevice if useGpu else "cpu") # get to know how to use both GPUs
    print(device)
    criteria = utils.getCriteria()
    
    if task == "train OCT": # train OCT or Fundus
        numClasses = len(criteria["All"]["OCT"])
        dbName = "ODADS/Data/Data/Train/OCT/"
        foldername = "O 2024-08-14 14-21-20 Transferred"
        wtsName = "O 2024-08-14 14-21-20 Transferred Best Epoch in 61 to 70.pth"
        batchSize = 16
        numEpochs = 30
        LR = 1e-3
        imgType = "O"
        usedPretrained = False
        now = datetime.now()
        filename = now.strftime(imgType + " %Y-%m-%d %H-%M-%S")
        trainClassify.train(filename, device, featureExtract, modelName, numClasses, batchSize, numEpochs, LR, usedPretrained, dbName, foldername, wtsName, imgType, crossValid = True)
        
    elif task == "train fundus": # train and fine-tune OCT
        numClasses = len(criteria["All"]["Fundus"])
        dbName = "ODADS/Data/Data/Train/Fundus/"
        foldername = "F 2024-08-14 14-21-54 Transferred"
        wtsName = "F 2024-08-14 14-21-54 Transferred Best Epoch in 41 to 50.pth"
        batchSize = 16
        numEpochs = 30
        LR = 1e-3
        imgType = "F"
        usedPretrained = False
        now = datetime.now()
        filename = now.strftime(imgType + " %Y-%m-%d %H-%M-%S")
        trainClassify.train(filename, device, featureExtract, modelName, numClasses, batchSize, numEpochs, LR, usedPretrained, dbName, foldername, wtsName, imgType, crossValid = True)
        
    elif task == 2: # train Fundus
        numClasses = len(criteria["All"]["Fundus"])
        dbName = "ODADS/Data/Data/Train/Fundus/"
        wtsName = ""
        batchSize = 16
        numEpochs = 100
        LR = 1e-3
        imgType = "F"
        usedPretrained = False
        now = datetime.now()
        foldername = now.strftime(imgType + " %Y-%m-%d %H-%M-%S")
        trainClassify.train(foldername, device, featureExtract, modelName, numClasses, batchSize, numEpochs, LR, usedPretrained, dbName, wtsName, imgType, crossValid = True)
        
    elif task == "classify OCT database": # gradCAM single image with OCT or Fundus
        numClasses = len(criteria["All"]["OCT"])
        dbName = "ODADS/Data/Data/Train/OCT/"
        foldername = "O 2024-08-15 12-32-19 Finetuning/"
        wtsName = "O 2024-08-15 12-32-19 Finetuning Best Epoch in 21 to 30.pth"
        classify.classifyDatabase(dbName, numClasses, device, featureExtract, modelName, foldername, wtsName)
        
    elif task == "get prob Fundus database": # gradCAM single image with OCT or Fundus
        numClasses = len(criteria["All"]["Fundus"])
        dbName = "ODADS/Data/Data/Train/Fundus/"
        foldername = "F 2024-08-15 12-32-17 Finetuning/"
        wtsName = "F 2024-08-15 12-32-17 Finetuning Best Epoch in 21 to 30.pth"
        classify.classifyDatabase(dbName, numClasses, device, featureExtract, modelName, foldername, wtsName)
        
    elif task == "train single disease": # train single disease
        now = datetime.now()
        dTime = now.strftime("%Y-%m-%d %H-%M-%S")
        oFoldername = "O 2024-08-15 12-32-19 Finetuning"
        oName = "O 2024-08-15 12-32-19 Finetuning Best Epoch in 21 to 30.pth"
        oClassSize = 5000
        fFoldername = "F 2024-08-15 12-32-17 Finetuning"
        fName = "F 2024-08-15 12-32-17 Finetuning Best Epoch in 21 to 30.pth"
        fClassSize = 3000
        numEpochs = 2
        dWtsName = ""
        gradeSize = 5000
        batchSize = 16
        diseaseName = "ERM"
        LR = 1e-3
        trainDiagnose.train(device, diseaseName, oFoldername, oName, oClassSize, fFoldername, fName, fClassSize, batchSize, gradeSize, numEpochs, LR, dWtsName, dTime)
        
    elif task == "train all diseases": # train all diseases
        now = datetime.now()
        dTime = now.strftime("%Y-%m-%d %H-%M-%S")
        oFoldername = "O 2024-08-15 12-32-19 Finetuning"
        oName = "Test_out O 2024-08-15 12-32-19 Finetuning Best Epoch in 21 to 30"
        oClassSize = 500
        fFoldername = "F 2024-08-15 12-32-17 Finetuning"
        fName = "Test_out F 2024-08-15 12-32-17 Finetuning Best Epoch in 21 to 30"
        fClassSize = 300
        numEpochs = 5
        dWtsName = ""
        gradeSize = 5000
        batchSize = 16
        LR = 1e-3
        for disease in criteria.keys():
            if disease == "All":
                continue
            trainDiagnose.train(device, disease, oFoldername, oName, oClassSize, fFoldername, fName, fClassSize, batchSize, gradeSize, numEpochs, LR, dWtsName, dTime)
            
    elif task == 6: # diagnose from images
        oImgPaths = []
        oImgs = [Image.open(imgPath) for imgPath in oImgPaths]
        fImgPaths = []
        fImgs = [Image.open(imgPath) for imgPath in fImgPaths]
        dWtsTime = "2024-01-01 00-00-00"
        oWts = "O 2024-05-09 22-42-15"
        fWts = "F 2024-08-07-16-48-56"
        diagnose.diagnose(oImgs, fImgs, diseaseName, device, modelName, dWtsTime, oWts, fWts)
    
    elif task == "test OCT": # test accuracy for a series of abnormity models
        numClasses = len(criteria["All"]["OCT"])
        dbName = "Test_out"
        dbPath = f"ODADS/Data/Data/{dbName}/OCT/"
        foldername = "O 2024-08-15 12-32-19 Finetuning"
        wtsName = "O 2024-08-15 12-32-19 Finetuning Best Epoch in 21 to 30.pth"
        testClassify.testAccWithLoader(device, featureExtract, modelName, numClasses, dbPath, dbName, foldername, wtsName)

    elif task == "test fundus": # test accuracy for a series of abnormity models
        numClasses = len(criteria["All"]["Fundus"])
        dbName = "Test_out"
        dbPath = f"ODADS/Data/Data/{dbName}/Fundus/"
        foldername = "F 2024-08-15 12-32-17 Finetuning"
        wtsName = "F 2024-08-15 12-32-17 Finetuning Best Epoch in 21 to 30.pth"
        testClassify.testAccWithLoader(device, featureExtract, modelName, numClasses, dbPath, dbName, foldername, wtsName)
        
    
if __name__ == "__main__":
    main()