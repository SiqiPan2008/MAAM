from Classify import classify
from Train import train
import torch

def main():
    string = ""
    modelName = "resnet"
    featureExtract = True
    useGpu = torch.cuda.is_available()
    print("CUDA available." if useGpu else
          "CUDA not available.")
    device = torch.device("cuda:0" if useGpu else "cpu") # get to know how to use both GPUs
    print(device)
    
    if string == "":
        train.train(device, featureExtract, modelName, 16, 15, 1e-3, True, "Fundus-normal-DR-selected", "F 2024-05-04 07-50-36", "F")
    else:
        numClasses = 2 # CHANGE NUM OF CLASSES!
        classify.classify(string, numClasses, device, useGpu, featureExtract, modelName, string)

if __name__ == "__main__":
    main()