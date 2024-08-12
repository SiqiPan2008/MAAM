from AbnormityModels import classify
from AbnormityModels import gradcam
from AbnormityModels import trainClassify
from DiagnosisModel import diagnose
from DiagnosisModel import trainDiagnose
import torch
from PIL import Image

def main():
    modelName = "resnet"
    featureExtract = True # True -> freeze conv layers; False -> train all layers
    useGpu = torch.cuda.is_available()
    print("CUDA available." if useGpu else
          "CUDA not available.")
    device = torch.device("cuda:0" if useGpu else "cpu") # get to know how to use both GPUs
    print(device)

    task = 2 # TASK!!!
    numClasses = 2 # CHANGE NUM OF CLASSES!
    if task == 0: # GAN
        # UNUSED
        batchSize = 1
        numEpochs = 5
        LR = 1e-5
        numWorkers = 4
        lambdaIdentity = 0.0
        lambdaCycle = 10
        normFolderName = "OCT-normal-ERM-fewshot/normal"
        abnormFolderName = "OCT-normal-ERM-fewshot/ERM"
        wtsName = ""
        abnormName = "ERM"
        dataType = "O"
        generate.generate(device, batchSize, numEpochs, LR, numWorkers, lambdaIdentity, lambdaCycle, normFolderName, abnormFolderName, wtsName, abnormName, dataType)
    elif task == 1: # train single OCT or Fundus
        dataset = "Fundus-normal-DR-demo/train"
        batchSize = 16
        epochs = 5
        LR = 1e-3
        imgType = "F"
        trainClassify.train(device, featureExtract, modelName, numClasses, batchSize, epochs, LR, True, dataset, "", imgType, crossValid = True)
    elif task == 2: # classify single image with OCT or Fundus
        string = "./Data/Fundus-normal-DR-selected/train/DR/16_left.jpeg"
        wts = "F 2024-08-07 16-48-56"
        img = Image.open(string)
        classify.classify(img, numClasses, device, featureExtract, modelName, wts)
    elif task == 3: # gradCAM single image with OCT or Fundus
        string = "./Data/OCT-normal-drusen-large/train/drusen/DRUSEN-303435-2.jpeg"
        wts = "O 2024-05-09 22-42-15"
        gradcam.highlight(string, numClasses, device, featureExtract, modelName, wts)
    elif task == 4: # train single disease
        oModelName = "./Data/Fundus-normal-DR-selected/train/DR/16_left.jpeg"
        fModelName = "O 2024-05-09 22-42-15"
        wts = ""
        gradeSize = 3000
        batchsize = 16
        dbName = ""
        diseaseName = "CSC"
        trainDiagnose.train(device, diseaseName, featureExtract, modelName, oModelName, fModelName, batchsize, gradeSize, numEpochs, LR, wts)
    elif task == 5: # train all diseases
        fModelName = "O 2024-05-09 22-42-15"
        gradeSize = 3000
        batchsize = 16
        dbName = "./Data"
        criteria = trainDiagnose.getCriteria()
        for disease in criteria.keys():
            wts = ""
            trainDiagnose.train(device, disease, featureExtract, modelName, oModelName, fModelName, batchsize, gradeSize, numEpochs, LR, dbName, wts)
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