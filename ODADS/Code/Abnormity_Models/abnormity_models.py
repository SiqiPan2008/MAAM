import os
from torch import nn
from torchvision import models
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def set_parameters_do_not_require_grad(model, featureExtract):
    if featureExtract:
        for param in model.parameters():
            param.requires_grad = False

def initialize_abnormity_model(net_name, num_classes, feature_extract, use_pretrained = True):
    model = None

    if net_name == "152":
        model = models.resnet152(weights = models.ResNet152_Weights.DEFAULT if use_pretrained else None)
    elif net_name == "101":
        model = models.resnet101(weights = models.ResNet101_Weights.DEFAULT if use_pretrained else None)
    elif net_name == "050":
        model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT if use_pretrained else None)
    elif net_name == "034":
        model = models.resnet34(weights = models.ResNet34_Weights.DEFAULT if use_pretrained else None)
    elif net_name == "018":
        model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT if use_pretrained else None)
    
    elif net_name == "019":
        model = models.vgg19(weights = models.VGG19_Weights.DEFAULT if use_pretrained else None)
    elif net_name == "016":
        model = models.vgg16(weights = models.VGG16_Weights.DEFAULT if use_pretrained else None)
    elif net_name == "013":
        model = models.vgg13(weights = models.VGG13_Weights.DEFAULT if use_pretrained else None)
    elif net_name == "011":
        model = models.vgg11(weights = models.VGG11_Weights.DEFAULT if use_pretrained else None)
    
    else:
        print("Invalid model name.")
        exit()
    
    set_parameters_do_not_require_grad(model, feature_extract if use_pretrained else False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))
    
    return model