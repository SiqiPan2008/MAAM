from Classify import classify
from Classify import gradcam
from TrainClassify import trainClassify
from TrainDiagnose import trainDiagnose
from Generate import generate
import torch

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
    if task == 0:
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
    elif task == 1:
        dataset = "Fundus-normal-DR-demo/train"
        batchSize = 16
        epochs = 5
        LR = 1e-3
        imgType = "F"
        trainClassify.train(device, featureExtract, modelName, numClasses, batchSize, epochs, LR, True, dataset, "", imgType, crossValid = True)
    elif task == 2:
        string = "./Data/Fundus-normal-DR-selected/train/DR/16_left.jpeg"
        wts = "F 2024-08-07 16-48-56"
        classify.classify(string, numClasses, device, featureExtract, modelName, wts)
    elif task == 3:
        string = "./Data/OCT-normal-drusen-large/train/drusen/DRUSEN-303435-2.jpeg"
        wts = "O 2024-05-09 22-42-15"
        gradcam.highlight(string, numClasses, device, featureExtract, modelName, wts)
    elif task == 4:
        oModelName = "./Data/Fundus-normal-DR-selected/train/DR/16_left.jpeg"
        fModelName = "O 2024-05-09 22-42-15"
        oNumClasses = 11
        fNumClasses = 9
        dNumClasses = 12
        wts = ""
        gradeSize = 3000
        batchsize = 16
        dbName = ""
        trainDiagnose.train(device, featureExtract, modelName, oModelName, fModelName, oNumClasses, dNumClasses, fNumClasses, batchsize, gradeSize, numEpochs, LR, wts)

if __name__ == "__main__":
    main()