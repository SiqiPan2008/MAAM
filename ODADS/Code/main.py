from AbnormityModels import classify, trainClassify, testClassify, gradcam
from DiagnosisModel import diagnose, trainDiagnose
from Utils import utils
import torch
from PIL import Image
from datetime import datetime
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    modelName = "resnet"
    cudaDevice = "cuda:1"
    featureExtract = True # True -> freeze conv layers; False -> train all layers
    useGpu = torch.cuda.is_available()
    print("CUDA available." if useGpu else
          "CUDA not available.")
    device = torch.device(cudaDevice if useGpu else "cpu") # get to know how to use both GPUs
    print(device)
    criteria = utils.getCriteria()
    
    task = 7
    
    if task == 0: # train OCT or Fundus
        numClasses = len(criteria["All"]["Fundus"])
        dbName = "ODADS/Data/Data/Old_train/Fundus/"
        wtsName = "F 2024-08-13 21-29-01"
        batchSize = 16
        numEpochs = 50
        LR = 1e-3
        imgType = "F"
        usedPretrained = False
        now = datetime.now()
        filename = now.strftime(imgType + " %Y-%m-%d %H-%M-%S")
        trainClassify.train(filename, device, featureExtract, modelName, numClasses, batchSize, numEpochs, LR, usedPretrained, dbName, wtsName, imgType, crossValid = True)
        
    elif task == 1: # train and fine-tune OCT
        numClasses = len(criteria["All"]["Fundus"])
        dbName = "ODADS/Data/Data/Train/Fundus/"
        wtsName = ""
        batchSize = 16
        numEpochs = 20
        LR = 1e-3
        imgType = "F"
        usedPretrained = False
        now = datetime.now()
        filename = now.strftime(imgType + " %Y-%m-%d %H-%M-%S")
        trainClassify.train(filename, device, featureExtract, modelName, numClasses, batchSize, numEpochs, LR, usedPretrained, dbName, wtsName, imgType, crossValid = True)
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
        
    elif task == 3: # gradCAM single image with OCT or Fundus
        numClasses = len(criteria["All"]["OCT"])
        string = "ODADS/Data/Data/Original/OCT/IntraretinalFluid/CNV-103044-13.jpeg"
        wts = "O 2024-08-13 11-44-08"
        gradcam.highlight(string, numClasses, device, featureExtract, modelName, wts)
        
    elif task == 4: # train single disease
        now = datetime.now()
        dTime = now.strftime("%Y-%m-%d %H-%M-%S")
        oModelName = "2024-05-09 22-42-15"
        fModelName = ""
        numEpochs = 200
        wts = ""
        gradeSize = 3000
        batchsize = 16
        dbName = ""
        diseaseName = "CSC"
        trainDiagnose.train(device, diseaseName, featureExtract, modelName, oModelName, fModelName, batchsize, gradeSize, numEpochs, LR, wts, dTime)
        
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
    
    elif task == 7: # test accuracy for a series of abnormity models
        numClasses = len(criteria["All"]["OCT"])
        dbName = "ODADS/Data/Data/Test/OCT/"
        foldername = "O 2024-08-14 14-21-20 Transferred"
        testClassify.testMultipleAcc(device, featureExtract, modelName, numClasses, dbName, foldername)

    elif task == 8: # test accuracy for a series of abnormity models
        numClasses = len(criteria["All"]["Fundus"])
        dbName = "ODADS/Data/Data/Test/Fundus/"
        foldername = "F 2024-08-14 14-21-54 Transferred"
        testClassify.testMultipleAcc(device, featureExtract, modelName, numClasses, dbName, foldername)
        
    
if __name__ == "__main__":
    main()