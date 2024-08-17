from Diagnosis_Model import diagnose_disease, train_diagnosis, test_diagnosis
from Abnormity_Models import test_abnormity, train_abnormity, classify_abnormity, gradcam
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
device = torch.device(cuda_device if use_gpu else "cpu")
print(device)

if task == "train all":
    train_abnormity.train(device, name + "TRRS - T")
    t_acc = test_abnormity.test_multiple(device, name + "TORS - T")
    train_abnormity.train(device, name + "TRRS - F")
    f_acc = test_abnormity.test_multiple(device, name + "TORS - F")
    test_abnormity.get_final_abnormity_model(name + "TRRS", t_acc, f_acc)
    test_abnormity.get_model_results(device, name + "TRMR")
    test_abnormity.get_model_results(device, name + "TOMR")
    
elif task == "train":
    if setting.is_abnormity(name):
        train_abnormity.train(device, name)
    elif setting.is_diagnosis1(name):
        train_diagnosis.train(device, name)
    elif setting.is_diagnosis2(name):
        train_diagnosis.train(device, name)
elif task == "test":
    if setting.is_abnormity(name):
        test_abnormity.test_multiple(device, name)
    elif setting.is_diagnosis1(name):
        test_diagnosis.test(device, name)
    elif setting.is_diagnosis2(name):
        test_diagnosis.test(device, name)
elif task == "get MR":
    if setting.is_abnormity(name):
        test_abnormity.get_model_results(device, name)