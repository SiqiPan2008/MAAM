from AbnormityModels import classify, gradcam
from DiagnosisModel import diagnose, trainDiagnose, testDiagnose
from ODADS.Code.AbnormityModels import test_abnormity, train_abnormity
from Utils import utils
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
        trainDiagnose.trainAbnormityNumModel(device, name)
    elif setting.is_diagnosis2(name):
        trainDiagnose.trainDiseaseProbModel(device, name)
elif task == "test":
    if setting.is_abnormity(name):
        test_abnormity.test_multiple_acc(device, name)
    elif setting.is_diagnosis1(name):
        testDiagnose.testAbnormityNumModel(device, name)
    elif setting.is_diagnosis2(name):
        testDiagnose.testDiseaseProbModel(device, name)
elif task == "get MR":
    if setting.is_abnormity(name):
        test_abnormity.get_model_results(device, name)
        
elif task == "":
    pass