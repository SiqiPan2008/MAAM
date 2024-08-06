import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels, isDownsampling = True, addActivation = True, **kwargs):
        super().__init__()
        if isDownsampling:
            self.conv = nn.Sequential(
                nn.Conv2d(inChannels, outChannels, padding_mode = "reflect", **kwargs), 
                # why are there two asterisks here? what is reflect?
                nn.InstanceNorm2d(outChannels), 
                # what is instancenorm2d?
                nn.ReLU(inplace = True) if addActivation else nn.Identity()
                # what is implace?
            )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(inChannels, outChannels, **kwargs),
                # transpose?
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
        # why plus?

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
        # why return tanh(x)?

def generate():
    a = 0