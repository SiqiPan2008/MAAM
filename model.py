from Classify import classify
from Train import train
from Generate import generate
import torch

def main():
    task = 0
    modelName = "resnet"
    featureExtract = True
    useGpu = torch.cuda.is_available()
    print("CUDA available." if useGpu else
          "CUDA not available.")
    device = torch.device("cuda:0" if useGpu else "cpu") # get to know how to use both GPUs
    print(device)

    numClasses = 4 # CHANGE NUM OF CLASSES!
    if task == 0:
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
        dataset = "OCT-normal-drusen-demo/train"
        batchSize = 16
        epochs = 5
        LR = 1e-3
        imgType = "O"
        train.train(device, featureExtract, modelName, numClasses, batchSize, epochs, LR, True, dataset, "", imgType, crossValid = 5)
    elif task == 2:
        string = "./Data/Fundus-normal-DR-selected/train/DR/16_left.jpeg"
        wts = "F 2024-05-05 18-09-32"
        classify.classify(string, numClasses, device, useGpu, featureExtract, modelName, wts)

if __name__ == "__main__":
    main()