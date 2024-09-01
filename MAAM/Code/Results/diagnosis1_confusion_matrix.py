import csv
import os
import numpy as np
from Utils import utils
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import time
import torch
import matplotlib.gridspec as gridspec
from Diagnosis_Model import diagnosis_model

def get_conf_mat(disease):
    setting = utils.get_setting()
    name = "000D1TOMR - " + disease
    label_levels = setting.get_disease_abnormity_num(disease, "All Abnormities") + 1
    mr_path = os.path.join(setting.D1_folder, name + ".bin")
    mr = np.fromfile(mr_path, dtype = np.float64).reshape((label_levels, setting.D1_test_class_size, label_levels))
    
    conf_mat = np.zeros((label_levels, label_levels), dtype=int)
    shape = mr.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            probs = mr[i][j]
            conf_mat[i][np.argmax(probs)] += 1
    return conf_mat

def plot_conf_mat():
    setting = utils.get_setting()
    diseases = setting.get_diseases(include_normal = False)
    folder_path = setting.D1_folder
    fig_folder = setting.fig_folder
    fig_file = "diagnosis1_confusion_matrix.pdf"
    fig_path = os.path.join(fig_folder, fig_file)
    
    fig = plt.figure(figsize = (8, 7))
    axs = fig.subplots(3, 4)
    fig.subplots_adjust(wspace = 0.2, hspace = 0.4)
    plt.rcParams['font.size'] = 9
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    row = 0
    col = 0
    
    with open(os.path.join(folder_path, "000D1TORS.csv"), "w", newline="") as file:  
        writer = csv.writer(file)
        for i in range(len(diseases)):
            disease = diseases[i]
            conf_mat = get_conf_mat(disease)
            levels = conf_mat.shape[0]
            fontsizes = {"2": 9,"3": 8,"4": 7, "5": 6, "6": 5}
            fontsize = fontsizes[str(levels)]
            corrects = sum([conf_mat[j][j] for j in range(conf_mat.shape[0])])
            total = sum([sum(conf_mat[j]) for j in range(conf_mat.shape[0])])
            acc = corrects / total
            writer.writerow([disease, corrects, total, acc])
            axs[row, col].set_title(setting.convert_disease_to_abbr(disease))
            axs[row, col].matshow(conf_mat, cmap='Blues')
            max = np.max(conf_mat)
            for i in range(conf_mat.shape[0]):
                for j in range(conf_mat.shape[1]):
                    axs[row, col].text(j, i, conf_mat[i, j], ha='center', va='center', color='black' if conf_mat[i, j] < max / 2 else 'white', fontsize = fontsize)
            axs[row, col].tick_params(axis='x', which='both', bottom=False)
            if col == 3:
                col = 0
                row += 1
            else:
                col += 1
        
    plt.savefig(fig_path, bbox_inches = "tight")
    plt.show()