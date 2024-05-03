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
    device = torch.device("cuda:1" if useGpu else "cpu") # get to know how to use both GPUs
    print(device)
    
    if string == "":
        train.train(device, featureExtract, modelName, 16, 25, 1e-3, True, "Fundus-normal-DR-selected", "", "F")
    else:
        numClasses = 2 # CHANGE NUM OF CLASSES!
        classify.classify(string, numClasses, device, useGpu, featureExtract, modelName, string + ".pth")

if __name__ == "__main__":
    main()