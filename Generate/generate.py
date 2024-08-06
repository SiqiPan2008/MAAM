import torch
import torch.nn as nn
from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels, isDownsampling = True, addActivation = True, **kwargs):
        super().__init__()
        if isDownsampling:
            self.conv = nn.Sequential(
                nn.Conv2d(inChannels, outChannels, padding_mode = "reflect", **kwargs), 
                ####### why are there two asterisks here? what is reflect?
                nn.InstanceNorm2d(outChannels), 
                ####### what is instancenorm2d?
                nn.ReLU(inplace = True) if addActivation else nn.Identity()
                ####### what is implace?
            )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(inChannels, outChannels, **kwargs),
                ####### transpose?
                nn.InstanceNorm2d(outChannels), 
                nn.ReLU(inplace = True) if addActivation else nn.Identity()
            )
    def forward(self, x):
        return self.conv(x)

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, addActivation = True, kernel_size = 3, padding = 1),
            ConvBlock(channels, channels, addActivation = False, kernel_size = 3, padding = 1)
        )
    def forward(self, x):
        return x + self.block(x)
        ####### why plus?

class Generator(nn.Module):
    def __init__(self, imgChannels, numFeatures = 64, numResiduals = 9):
        super().__init__()
        self.initialLayer = nn.Sequential(
            nn.Conv2d(imgChannels, numFeatures, kernel_size = 7, stride = 1, padding = 3, padding_mode = "reflect"),
            nn.ReLU(inplace = True)
        )
        self.downsamplingLayers = nn.ModuleList(
            [
                ConvBlock(numFeatures, numFeatures * 2, isDownsampling = True, kernel_size = 3, stride = 2, padding = 1),
                ConvBlock(numFeatures * 2, numFeatures * 4, isDownsampling = True, kernel_size = 3, stride = 2, padding = 1)
            ]
        )
        self.residualLayers = nn.Sequential(
            *[ResBlock(numFeatures * 4) for _ in range(numResiduals)]
        )
        self.upsamplingLayers = nn.ModuleList(
            [
                ConvBlock(numFeatures * 4, numFeatures * 2, isDownsampling = False, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                ConvBlock(numFeatures * 2, numFeatures, isDownsampling = False, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
            ]
        )
        self.lastLayer = nn.Conv2d(numFeatures, imgChannels, kernel_size = 7, stride = 1, padding = 3, padding_mode = "reflect")
    def forward(self, x):
        x = self.initialLayer(x)
        for layer in self.downsamplingLayers:
            x = layer(x)
        x = self.residualLayers(x)
        for layer in self.upsamplingLayers:
            x = layer(x)
        x = self.lastLayer(x)
        return torch.tanh(x)
        ####### why return tanh(x)?

class ConvInstanceNormLeakyReLUBlock(nn.Module):
    ####### what is this for?
    def __init__(self, inChannels, outChannels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernel_size = 4, stride = stride, padding = 1, bias = True, padding_mode = "reflect"),
            ####### what is bias?
            nn.InstanceNorm2d(outChannels),
            nn.LeakyReLU(0.2, inplace = True)
            ####### why leaky? read paper
        )
    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, inChannels = 3, features = [64, 128, 256, 512]):
        super().__init__()
        self.initialLayer = nn.Sequential(
            nn.Conv2d(inChannels, features[0], kernel_size = 4, stride = 2, padding = 1, padding_mode = "reflect"),
            nn.LeakyReLU(0.2, inplace = True)
        )
        ###### difference from ConvInstanceNormLeakyReLUBlock(inChannels, features[0], 2)? is it the bias=True?
        layers = []
        inChannels = features[0]
        for feature in features[1:]:
            layers.append(ConvInstanceNormLeakyReLUBlock(inChannels, feature, stride = 1 if feature == features[-1] else 2))
            inChannels = feature
        layers.append(nn.Conv2d(inChannels, 1, kernel_size = 4, stride = 1, padding = 1, padding_mode = "reflect"))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        x = self.initialLayer(x)
        x = self.model(x)
        return torch.sigmoid(x)

class NormalDiseaseDataset(Dataset):
    def __init__(self, normDir, abnormDir, transform = None):
        self.normDir = normDir
        self.abnormDir = abnormDir
        self.transform = transform
        self.normImgs = os.listdir(normDir)
        self.abnormImgs = os.listdir(abnormDir)
        self.normLen = len(self.normImgs)
        self.abnormLen = len(self.abnormImgs)
        self.datasetLen = max(self.normLen, self.abnormLen)
    def __len__(self):
        return self.datasetLen
    def __getitem__(self, index):
        normImgName = self.normImgs[index % self.normLen]
        abnormImgName = self.abnormImgs[index % self.abnormLen]
        normPath = os.path.join(self.normDir, normImgName)
        abnormPath = os.path.join(self.abnormDir, abnormImgName)
        normImg = np.array(Image.open(normPath).convert("RGB"))
        abnormImg = np.array(Image.open(abnormPath).convert("RGB"))
        if self.transform:
            augmentations = self.transform(image = abnormImg, image0 = normImg)
            normImg, abnormImg = augmentations["image0"], augmentations["image"]
        return abnormImg, normImg

def generate():
    a = 0