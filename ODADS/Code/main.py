from Diagnosis_Model import train_diagnosis, test_diagnosis
from Abnormity_Models import test_abnormity, train_abnormity
from Results import \
    abnormity_loss_and_acc, abnormity_gradCAM, abnormity_ROC, abnormity_tSNE, abnormity_confusion_matrix, \
    diagnosis1_acc_barchart, diagnosis1_confusion_matrix, diagnosis2_tSNE_ROC_confusion_matrix
from Utils import utils
import torch
import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

setting = utils.get_setting()

task = sys.argv[1]
name = sys.argv[2]
cuda_device = sys.argv[3]

use_gpu = torch.cuda.is_available()
device = torch.device(cuda_device if use_gpu else "cpu")
print(device)


if task == "get results":
    if name == "A: Loss and Acc":
        abnormity_loss_and_acc.plot_loss_and_acc()
    elif name == "A: ROC":
        abnormity_ROC.plot_roc()
    elif name == "A: tSNE":
        abnormity_tSNE.plot_tsne()
    elif name == "A: confusion matrix":
        abnormity_confusion_matrix.plot_all_conf_mat()
    elif name == "A: gradCAM":
        abnormity_gradCAM.highlight(device, sys.argv[4], sys.argv[5])
    elif name == "D1: Acc barchart":
        diagnosis1_acc_barchart.plot_acc_barchart()
    elif name == "D1: confusion matrix":
        diagnosis1_confusion_matrix.plot_conf_mat()
    elif name == "D2: tSNE, ROC, confusion matrix":
        diagnosis2_tSNE_ROC_confusion_matrix.plot_tSNE_ROC_conf_mat(plt_tSNE = True, plt_ROC = True, plt_conf_mat = True)
    elif task == "graph all":
        abnormity_confusion_matrix.plot_all_conf_mat()
        diagnosis1_confusion_matrix.plot_conf_mat()
        
        abnormity_loss_and_acc.plot_loss_and_acc()
        abnormity_ROC.plot_roc()
        abnormity_tSNE.plot_tsne()
        diagnosis1_acc_barchart.plot_acc_barchart()
        diagnosis1_confusion_matrix.plot_conf_mat()



elif task == "abnormity all":
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
    train_diagnosis.train(device, [name + "D1TRRS" for name in names])
    train_diagnosis.train(device, [name + "D2TRRS" for name in names])
    test_diagnosis.test(device, [name + "D1TORS" for name in names])
    test_diagnosis.test(device,[name + "D2TORS" for name in names])


else:
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