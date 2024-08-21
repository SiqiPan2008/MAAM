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

# multi-task commands
if task == "abnormity all":
    train_abnormity.train(device, name + "TRRS - T")
    test_abnormity.test_multiple(device, name + "TORS - T")
    test_abnormity.get_best_abnormity_model(name + "TORS - T")
    train_abnormity.train(device, name + "TRRS - F")
    test_abnormity.test_multiple(device, name + "TORS - F")
    test_abnormity.get_best_abnormity_model(name + "TORS - F")
    test_abnormity.choose_t_or_f_abnormity_model(name + "TRRS")
    test_abnormity.get_model_results(device, name + "TRMR")
    test_abnormity.get_model_results(device, name + "TOMR")

elif task == "diagnosis all":
    names = name.split(", ")
    #train_diagnosis.train(device, [name + "D1TRRS" for name in names])
    #train_diagnosis.train(device, [name + "D2TRRS" for name in names])
    test_diagnosis.test(device, [name + "D1TORS" for name in names])
    test_diagnosis.test(device,[name + "D2TORS" for name in names])

# single-task commands
if setting.is_abnormity(name):
    if task == "train":
        train_abnormity.train(device, name)
    elif task == "test":
        test_abnormity.test_multiple(device, name)
    elif task == "get MR":
        test_abnormity.get_model_results(device, name)
elif setting.is_diagnosis1(name) or setting.is_diagnosis2(name):
    # net_name_f
    names = name.split(", ")
    if task == "train":
        train_diagnosis.train(device, names)
    elif task == "test":
        test_diagnosis.test(device, names)