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
        #wtsName = filename
        #numEpochs = 30
        #usedPretrained = False
        #now = datetime.now()
        #fineTuneFilename = now.strftime(imgType + " %Y-%m-%d %H-%M-%S")
        #trainClassify.train(fineTuneFilename, device, featureExtract, modelName, numClasses, batchSize, numEpochs, LR, usedPretrained, dbName, wtsName, imgType, crossValid = True)
        
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
        oModelName = "O 2024-08-15 12-32-19 Finetuning/O 2024-08-15 12-32-19 Finetuning Best Epoch in 21 to 30.pth"
        fModelName = "F 2024-08-15 12-32-17 Finetuning/F 2024-08-15 12-32-17 Finetuning Best Epoch in 21 to 30.pth"
        numEpochs = 2
        wts = ""
        gradeSize = 5000
        batchsize = 16
        dbName = ""
        diseaseName = "ERM"
        LR = 1e-3
        dbName = "ODADS/Data/Data/Train/"
        trainDiagnose.train(device, diseaseName, featureExtract, modelName, oModelName, fModelName, batchsize, gradeSize, numEpochs, LR, dbName, wts, dTime)
        
    elif task == 5: # train all diseases
        now = datetime.now()
        dTime = now.strftime("%Y-%m-%d %H-%M-%S")
        fModelName = "O 2024-05-09 22-42-15"
        gradeSize = 3000
        batchsize = 16
        dbName = "./Data"
        criteria = trainDiagnose.utils.getCriteria()
        for disease in criteria.keys():
            wts = ""
            trainDiagnose.train(device, disease, featureExtract, modelName, oModelName, fModelName, batchsize, gradeSize, numEpochs, LR, dbName, wts, dTime)
            
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
        dbName = "ODADS/Data/Data/Train/OCT/"
        foldername = "O 2024-08-15 12-32-19 Finetuning"
        wtsName = "O 2024-08-15 12-32-19 Finetuning Best Epoch in 21 to 30.pth"
        testClassify.testAccWithLoader(device, featureExtract, modelName, numClasses, dbName, foldername, wtsName)

    elif task == "test fundus": # test accuracy for a series of abnormity models
        pass
        
    
if __name__ == "__main__":
    outputs = np.fromfile("ODADS/Data/Results/F 2024-08-15 12-32-17 Finetuning/F 2024-08-15 12-32-17 Finetuning Best Epoch in 21 to 30.pth.bin", dtype = np.float64)
    outputs = outputs.reshape((9, 3000, 9))
    print(outputs)
    main()