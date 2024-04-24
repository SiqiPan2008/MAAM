import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models, datasets
import imageio
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def trainModel():
    data_dir = "../data"
    train_dir = data_dir + "/train"
    test_dir = data_dir + "/test"
    batch_size = 16
    image_datasets = {}
    return