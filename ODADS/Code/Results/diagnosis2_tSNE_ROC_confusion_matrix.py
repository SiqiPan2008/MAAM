import os
import torch
import time
import random
import numpy as np
from Utils import utils
import csv
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from Diagnosis_Model import diagnosis_model
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def plot_tSNE(features, labels, diseases):
    setting = utils.get_setting()
    fig_folder = setting.fig_folder
    fig_file = "diagnosis2_tSNE.pdf"
    fig_path = os.path.join(fig_folder, fig_file)
    
    fig = plt.figure(figsize = (9, 8))
    
    tsne = TSNE(n_components=2)
    features_tsne = tsne.fit_transform(np.array(features))
    
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap='jet', alpha=0.5)
    plt.colorbar(scatter)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    
    plt.savefig(fig_path)
    plt.show()

def plot_all_ROC(labels, probs, diseases, draw_roc = True):
    setting = utils.get_setting()
    fig_folder = setting.fig_folder
    fig_file = "diagnosis2_ROC.pdf"
    fig_path = os.path.join(fig_folder, fig_file)
    
    if draw_roc:
        fig = plt.figure(figsize = (8, 8))
        axs = fig.subplots(4, 4)
    
    row = 0
    col = 0
    aucs = []
    
    for disease in diseases:
        fpr, tpr, _ = roc_curve(labels[disease], probs[disease])
        aucs.append(auc(fpr, tpr))
        if draw_roc:
            axs[row, col].plot(fpr, tpr, color = "red")
            axs[row, col].set_xlim([0.0, 1.0])
            axs[row, col].set_ylim([0.0, 1.0])
            axs[row, col].plot([0, 1], [0, 1], color = "black")
        if col == 3:
            col = 0
            row += 1
        else:
            col += 1
    
    for col in range(1, 4):
        axs[3, col].set_axis_off()
    
    if draw_roc:
        plt.savefig(fig_path)
        plt.show()
    return aucs

def get_single_conf_mat(complete_conf_mat, index):
    conf_mat = np.zeros((2, 2), dtype=int)
    num = complete_conf_mat.shape[0]
    for i in range(num):
        for j in range(num):
            conf_mat[1 if i == index else 0, 1 if j == index else 0] += complete_conf_mat[i, j]
    return conf_mat

def plot_conf_mat(conf_mat, diseases):
    setting = utils.get_setting()
    fig_folder = setting.fig_folder
    fig_file = "diagnosis2_confusion_matrix.pdf"
    fig_path = os.path.join(fig_folder, fig_file)
    
    mat = plt.matshow(conf_mat, cmap='Blues')
    max = np.max(conf_mat)
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            plt.text(j, i, conf_mat[i, j], ha='center', va='center', color='black' if conf_mat[i, j] < max / 2 else 'white')
    plt.colorbar(mat)
    
    plt.savefig(fig_path)
    plt.show()

# also calculates numerical results
def plot_tSNE_ROC_conf_mat(plt_tSNE = True, plt_ROC = True, plt_conf_mat = True):
    setting = utils.get_setting()
    name = "000D2TOMR"
    tSNE_point_num = setting.tSNE_point_num
    diseases_including_normal = setting.get_diseases(include_normal = True)
    disease_num_including_normal = setting.get_disease_num(include_normal = True)
    mr_path = os.path.join(setting.D2_folder, name + ".bin")
    mr = np.fromfile(mr_path, dtype = np.float64).reshape((disease_num_including_normal, setting.D2_test_class_size, disease_num_including_normal))
    
    roc_labels = {disease: [] for disease in diseases_including_normal}
    roc_probs = {disease: [] for disease in diseases_including_normal}
    tSNE_labels = [[] for _ in diseases_including_normal]
    conf_mat = np.zeros((disease_num_including_normal, disease_num_including_normal), dtype=int)
    
    start_time = time.time()
    corrects = 0
    total = 0
    shape = mr.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            output = mr[i][j]
            predicted = np.argmax(np.array(output))
            corrects += (predicted == i).item()
            total += 1
            
            conf_mat[i][predicted] += 1
            for k in range(len(diseases_including_normal)):
                disease = diseases_including_normal[i]
                roc_labels[disease].append(k == i)
                roc_probs[disease].append(output[k])
            
    acc = corrects / total
    time_elapsed = time.time() - start_time
    print(f"Time elapsed {time_elapsed // 60 :.0f}m {time_elapsed % 60 :.2f}s")
    print(f"acc: {corrects}/{total} = {acc :.4f}")
    
    tSNE_features = [[] for _ in range(disease_num_including_normal)]
    for i in range(len(tSNE_features)):
        #tSNE_features[i] = np.random.choice(tSNE_features[i], tSNE_point_num)
        tSNE_features[i] = mr[i][np.random.choice(mr[i].shape[0], size=tSNE_point_num, replace=False)]
        tSNE_labels[i] = [i for _ in range(tSNE_point_num)]
    tSNE_features = np.concatenate(tSNE_features)
    tSNE_labels = np.concatenate(tSNE_labels)
    
    if plt_tSNE:
        plot_tSNE(tSNE_features, tSNE_labels, diseases_including_normal)
    if plt_ROC:
        aucs = plot_all_ROC(roc_labels, roc_probs, diseases_including_normal)
    if plt_conf_mat:
        plot_conf_mat(conf_mat, diseases_including_normal)
    
    with open(os.path.join(setting.D2_folder, "000D2TORS.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([corrects, total, acc])
    
    with open(os.path.join(setting.table_folder, "diagnosis2.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        for i in range(len(diseases_including_normal)):
            disease_conf_mat = get_single_conf_mat(conf_mat, i)
            tp = disease_conf_mat[0][0]
            fn = disease_conf_mat[0][1]
            fp = disease_conf_mat[1][0]
            tn = disease_conf_mat[1][1]
            precision = tp / (tp + fp)
            specificity = tn / (tn + fp)
            sensitivity = tp / (tp + fn)
            f1 = 2 * precision * sensitivity / (precision + sensitivity)
            writer.writerow([
                diseases_including_normal[i],
                precision,
                sensitivity,
                specificity,
                f1, 
                aucs[i]
            ])