from Classify import classify
from Train import train
import torch

def main():
    string = "" # "./Data/Fundus-normal-DR-selected/train/DR/16_left.jpeg"
    modelName = "resnet"
    featureExtract = True
    useGpu = torch.cuda.is_available()
    print("CUDA available." if useGpu else
          "CUDA not available.")
    device = torch.device("cuda:0" if useGpu else "cpu") # get to know how to use both GPUs
    print(device)

    numClasses = 4 # CHANGE NUM OF CLASSES!
    if string == "":
        train.train(device, featureExtract, modelName, numClasses, 16, 5, 1e-3, True, "OCT-normal-drusen-demo/train", "", "O", crossValid = 5)
    else:
        classify.classify(string, numClasses, device, useGpu, featureExtract, modelName, "F 2024-05-05 18-09-32")

if __name__ == "__main__":
    main()