from RunModel import runModel
from GiveResult import GiveResult
import torch

def main():
    string = input("Filename (empty for training): ")
    
    modelName = "resnet"
    featureExtract = True
    useGpu = torch.cuda.is_available()
    print("CUDA available." if useGpu else "CUDA not available.")
    device = torch.device("cuda:0" if useGpu else "cpu") # get to know how to use both GPUs
    print(device)
    fileName = "trainedModel.pth"
    
    if string == "":
        runModel.runModel(device, featureExtract, modelName, fileName)
    else:
        numClasses = 2 # CHANGE NUM OF CLASSES!
        GiveResult.giveResult(string, numClasses, device, useGpu, featureExtract, modelName, fileName)
    
    

if __name__ == "__main__":
    main()