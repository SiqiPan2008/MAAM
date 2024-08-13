from AbnormityModels import classify, trainClassify, gradcam
from DiagnosisModel import diagnose, trainDiagnose
from Utils import utils
import torch
from PIL import Image
import datetime
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    modelName = "resnet"
    featureExtract = True # True -> freeze conv layers; False -> train all layers
    useGpu = torch.cuda.is_available()
    print("CUDA available." if useGpu else
          "CUDA not available.")
    device = torch.device("cuda:0" if useGpu else "cpu") # get to know how to use both GPUs
    print(device)
    criteria = utils.getCriteria()
    
    task = 1 
    numClasses = len(criteria["All"]["OCT"])
    
    if task == 1: # train OCT or Fundus
        dbName = "ODADS/Data/Data/Transformed/OCT/"
        wtsName = "O 2024-08-13 08-07-02"
        batchSize = 16
        numEpochs = 17
        LR = 1e-3
        imgType = "O"
        usedPretrained = False
        trainClassify.train(device, featureExtract, modelName, numClasses, batchSize, numEpochs, LR, usedPretrained, dbName, wtsName, imgType, crossValid = True)
        
    elif task == 2: # classify single image with OCT or Fundus
        string = "ODADS/Data/Data/Original/OCT"
        wts = ""
        img = Image.open(string)
        classify.classify(img, numClasses, device, featureExtract, modelName, wts)
        
    elif task == 3: # gradCAM single image with OCT or Fundus
        string = "./Data/OCT-normal-drusen-large/train/drusen/DRUSEN-303435-2.jpeg"
        wts = "O 2024-05-09 22-42-15"
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
    

if __name__ == "__main__":
    main()