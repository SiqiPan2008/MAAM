import torch
import torch.nn as nn
from PIL import Image
import os
import torch.utils
from torch.utils.data import Dataset
import numpy as np
import torch.utils.data
from Train import train
from torchvision import transforms
from datetime import datetime
import sys
import random
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image

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

class NormAbnormDataset(Dataset):
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
            normImg = self.transform(normImg)
            abnormImg = self.transform(abnormImg)
        return abnormImg, normImg

def saveCheckpoint(model, optimizer, filename):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, filename)
    
def loadCheckpoint(model, device, checkpointFile, optimizer, lr):
    checkpoint = torch.load(checkpointFile, map_location = device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for paramGroup in optimizer.param_groups:
        paramGroup["lr"] = lr

'''
def seedAll(seed = 47):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
'''






def trainCycleGAN(gNorm, gAbnorm, dNorm, dAbnorm, loader, dOptimizer, gOptimizer, l1, mse, dScaler, gScaler, device, lambdaIdentity, lambdaCycle, foldername):
    normReals = 0
    normFakes = 0
    ####### why no abnormReals and abnormFakes?
    loop = tqdm(loader, leave = True)
    
    for x in enumerate(loop):
        print(x)
    
    for index, (norm, abnorm) in enumerate(loop):
        norm = norm.to(device)
        abnorm = abnorm.to(device)
        
        with torch.cuda.amp.autocast():
            fakeNorm = gNorm(abnorm)
            dNormReal = dNorm(norm)
            dNormFake = dNorm(fakeNorm.detach())
            ####### what is detach()
            normReals += dNormReal.mean().item()
            normFakes += dNormFake.mean().item()
            dNormRealLoss = mse(dNormReal, torch.ones_like(dNormReal))
            dNormFakeLoss = mse(dNormFake, torch.zeros_like(dNormFake))
            dNormLoss = dNormRealLoss + dNormFakeLoss
            
            fakeAbnorm = gAbnorm(norm)
            dAbnormReal = dAbnorm(abnorm)
            dAbnormFake = dAbnorm(fakeAbnorm.detach())
            dAbnormRealLoss = mse(dAbnormReal, torch.ones_like(dAbnormReal))
            dAbnormFakeLoss = mse(dAbnormFake, torch.zeros_like(dAbnormFake))
            dAbnormLoss = dAbnormRealLoss + dAbnormFakeLoss
            
            dLoss = (dNormLoss + dAbnormLoss) / 2
        
        dOptimizer.zero_grad()
        dScaler.scale(dLoss).backward()
        dScaler.step(dOptimizer)
        dScaler.update()
        
        with torch.cuda.amp.autocast():
            # adversarial losses
            dNormFake = dNorm(fakeNorm)
            dAbnormFake = dAbnorm(fakeAbnorm)
            gNormLoss = mse(dNormFake, torch.ones_like(dNormFake))
            gAbnormLoss = mse(dAbnormFake, torch.ones_like(dAbnormFake))
            
            # cycle losses
            cycleNorm = gNorm(fakeAbnorm)
            cycleAbnorm = gAbnorm(fakeNorm)
            cycleNormLoss = l1(norm, cycleNorm)
            cycleAbnormLoss = l1(abnorm, cycleAbnorm)
            
            # identity losses
            # identityNorm = gNorm(norm)
            # identityAbnorm = gAbnorm(abnorm)
            # identityNormLoss = l1(norm, identityNorm)
            # identityAbnormLoss = l1(abnorm, identityAbnorm)
            
            gLoss = (
                (gNormLoss + gAbnormLoss)
                + lambdaCycle * (cycleNormLoss + cycleAbnormLoss)
                # + lambdaIdentity * (identityNormLoss + identityAbnormLoss)
            )
            
        gOptimizer.zero_grad()
        gScaler.scale(gLoss).backward()
        gScaler.step(gOptimizer)
        gScaler.update()
        
        save_image(fakeAbnorm, f"GeneratedImg/{foldername}/{index}.png")
        






def generate(device, batchSize, numEpochs, lr, numWorkers, lambdaIdentity, lambdaCycle, normFoldername, abnormFoldername, wtsName, abnormName, dataType):
    # wtsName: name of folder of weights, e.g. "GAN O MH 2024-08-06 12-50-24"
    # abnormName: name of abnormity, e.g. "MH"
    # dataType: OCT "O" or fundus "F"
    now = datetime.now()
    timeStr = now.strftime("%Y-%m-%d %H-%M-%S")
    foldername = f"GAN {dataType} {abnormName} {timeStr}"
    log = f"Log/{foldername}.csv"
    gNormCheckpoint = f"TrainedModel/{foldername}/{foldername} G Norm.pth"
    gAbnormCheckpoint = f"TrainedModel/{foldername}/{foldername} G Abnorm.pth"
    dNormCheckpoint = f"TrainedModel/{foldername}/{foldername} D Norm.pth"
    dAbnormCheckpoint = f"TrainedModel/{foldername}/{foldername} D Abnorm.pth"
    
    dataTransforms = transforms.Compose(
        [
        transforms.Lambda(lambda img: train.resizeLongEdge(img, longEdgeSize = 256)),
        transforms.ToTensor()
        ]
    )
    
    dNorm = Discriminator(inChannels = 3).to(device)
    dAbnorm = Discriminator(inChannels = 3).to(device)
    gNorm = Generator(imgChannels = 3, numResiduals = 9).to(device)
    gAbnorm = Generator(imgChannels = 3, numResiduals = 9).to(device)
    dOptimizer = optim.Adam(
        list(dNorm.parameters()) + list(dAbnorm.parameters()), 
        lr = lr,
        betas = (0.5, 0.999)
        ####### what are betas?
    )
    gOptimizer = optim.Adam(
        list(gNorm.parameters()) + list(gAbnorm.parameters()), 
        lr = lr,
        betas = (0.5, 0.999)
    )
    
    L1 = nn.L1Loss()
    mse = nn.MSELoss()
    ####### what are these?
    
    if wtsName:
        loadCheckpoint(dNorm, device, dNormCheckpoint, dOptimizer, lr)
        loadCheckpoint(dAbnorm, device, dAbnormCheckpoint, dOptimizer, lr)
        loadCheckpoint(gNorm, device, gNormCheckpoint, gOptimizer, lr)
        loadCheckpoint(gAbnorm, device, gAbnormCheckpoint, gOptimizer, lr)
    
    trainData = NormAbnormDataset(f"Data/{normFoldername}", f"Data/{abnormFoldername}", transform = transforms)
    trainLoader = torch.utils.data.DataLoader(trainData, batch_size = batchSize, shuffle = True, num_workers = numWorkers, pin_memory = True)
    gScaler = torch.cuda.amp.GradScaler()
    dScaler = torch.cuda.amp.GradScaler()
    ####### where is validData used?
    
    for epoch in range(numEpochs):
        trainCycleGAN(gNorm, gAbnorm, dNorm, dAbnorm, trainLoader, dOptimizer, gOptimizer, L1, mse, dScaler, gScaler, device, lambdaIdentity, lambdaCycle, foldername)
    
    saveCheckpoint(gNorm, gOptimizer, filename = gNormCheckpoint)
    saveCheckpoint(gAbnorm, gOptimizer, filename = gAbnormCheckpoint)
    saveCheckpoint(dNorm, dOptimizer, filename = dNormCheckpoint)
    saveCheckpoint(dAbnorm, dOptimizer, filename = dAbnormCheckpoint)