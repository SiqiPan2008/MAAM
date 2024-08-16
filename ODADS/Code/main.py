from ODADS.Code.Diagnosis_Model import diagnose_disease, train_diagnose, test_diagnose
from ODADS.Code.Abnormity_Models import test_abnormity, train_abnormity, classify_abnormity, gradcam
from ODADS.Code.Utils import utils
import torch
import numpy as np
from PIL import Image
from datetime import datetime
import os
import sys
import json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

setting = utils.get_setting()

task = sys.argv[1]
name = sys.argv[2]
cuda_device = sys.argv[3]

use_gpu = torch.cuda.is_available()
device = torch.device(cuda_device if use_gpu else "cpu") # get to know how to use both GPUs
print(device)

if task == "train":
    if setting.is_abnormity(name):
        train_abnormity.train(device, name)
    elif setting.is_diagnosis1(name):
        train_diagnose.trainAbnormityNumModel(device, name)
    elif setting.is_diagnosis2(name):
        train_diagnose.trainDiseaseProbModel(device, name)
elif task == "test":
    if setting.is_abnormity(name):
        test_abnormity.test_multiple_acc(device, name)
    elif setting.is_diagnosis1(name):
        train_diagnose.testAbnormityNumModel(device, name)
    elif setting.is_diagnosis2(name):
        train_diagnose.testDiseaseProbModel(device, name)
elif task == "get MR":
    if setting.is_abnormity(name):
        test_abnormity.get_model_results(device, name)
        
elif task == "":
    pass