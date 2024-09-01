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
from matplotlib.lines import Line2D
from Diagnosis_Model import diagnosis_model
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def plot_tSNE(features, labels, diseases):
    plt.rcParams['axes.titlesize'] = 9
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    
    setting = utils.get_setting()
    fig_folder = setting.fig_folder
    fig_file = "diagnosis2_tSNE.pdf"
    fig_path = os.path.join(fig_folder, fig_file)
    
    fig = plt.figure(figsize = (4, 3))
    
    tsne = TSNE(n_components=2)
    features_tsne = tsne.fit_transform(np.array(features))
    s = 2
    plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap='jet', marker='o', s=s)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    
    handles = []
    for i, disease in enumerate(diseases):
        handle = Line2D([], [], color=plt.get_cmap('jet')(i / len(diseases)), marker='o', linestyle='None', markersize=s, label=disease)
        handles.append(handle)
    plt.legend(handles=handles, bbox_to_anchor=(1, 0.5), loc='center left')
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.show()

def plot_all_ROC(labels, probs, diseases, draw_roc = True):
    plt.rcParams['axes.titlesize'] = 9
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    
    setting = utils.get_setting()
    fig_folder = setting.fig_folder
    fig_file = "diagnosis2_ROC.pdf"
    fig_path = os.path.join(fig_folder, fig_file)
    
    if draw_roc:
        fig = plt.figure(figsize = (8, 5))
        axs = fig.subplots(3, 5)
        plt.subplots_adjust(wspace=0.1, hspace=0.125)
    
    row = 0
    col = 0
    aucs = []
    
    for disease in diseases:
        fpr, tpr, _ = roc_curve(labels[disease], probs[disease])
        aucs.append(auc(fpr, tpr))
        if draw_roc:
            ax = axs[row, col]
            abbr = setting.convert_disease_to_abbr(disease)
            ax.plot(fpr, tpr, color = "#FF0000", linewidth = 1.2)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.0])
            ax.plot([0, 1], [0, 1], color = "gray", linewidth = 0.6)
            ax.text(0.95, 0.05, abbr, verticalalignment='bottom', horizontalalignment='right', fontsize=9)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax.xaxis.set_ticks_position('top')
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            ax.xaxis.set_tick_params(labelbottom=False, labeltop=True)
            if row == 0:
                ax.set_xticklabels(["0", "", "0.5", "", "1"])
            else:
                ax.set_xticks([])
            if col == 0:
                ax.set_yticklabels(["0", "", "0.5", "", "1"])
            else:
                ax.set_yticks([])
        if col == 4:
            col = 0
            row += 1
        else:
            col += 1
    
    for col in range(3, 5):
        axs[2, col].set_axis_off()

    if draw_roc:
        legend_lines = [
            Line2D([0], [0], color="#FF0000", linewidth = 1.2, linestyle='-', label='ROC'),
            Line2D([0], [0], color="gray", linewidth = 0.6,linestyle='-', label='y = x'),
        ]
        fig.legend(handles=legend_lines, ncols = 2, loc = "lower right", bbox_to_anchor = (0.91,0.1))
        fig.text(0.898, 0.18, "Horizontal axes: specificity\nVertical axes: sensitivity", fontsize=9, 
            verticalalignment='bottom', horizontalalignment='right', bbox=dict(facecolor='white', edgecolor='#D1D1D1', boxstyle='round,pad=0.25'))
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
    plt.rcParams['axes.titlesize'] = 9
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    
    setting = utils.get_setting()
    fig_folder = setting.fig_folder
    fig_file = "diagnosis2_confusion_matrix.pdf"
    fig_path = os.path.join(fig_folder, fig_file)
    

    mat = plt.matshow(conf_mat, cmap='Blues')
    max = np.max(conf_mat)
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            plt.text(j, i, conf_mat[i, j], ha='center', va='center', color='black' if conf_mat[i, j] < max / 2 else 'white', fontsize=7)
    plt.xticks(np.arange(len(diseases)), labels=diseases, rotation=-90)
    plt.yticks(np.arange(len(diseases)), labels=diseases)
    plt.tick_params(axis='x', which='both', bottom=False)
    plt.savefig(fig_path)
    plt.show()

# also calculates numerical results
def plot_tSNE_ROC_conf_mat(plt_tSNE = True, plt_ROC = True, plt_conf_mat = True):
    setting = utils.get_setting()
    name = "000D2TOMR"
    tSNE_point_num = setting.tSNE_point_num
    diseases_including_normal = setting.get_diseases(include_normal = True)
    abbr_diseases = setting.get_abbr_diseases(include_normal = True)
    disease_num_including_normal = setting.get_disease_num(include_normal = True)
    mr_path = os.path.join(setting.D2_folder, name + ".bin")
    mr = np.fromfile(mr_path, dtype = np.float64).reshape((disease_num_including_normal, setting.D2_test_class_size, disease_num_including_normal))
    
    roc_labels = {disease: [] for disease in diseases_including_normal}
    roc_probs = {disease: [] for disease in diseases_including_normal}
    tSNE_labels = [[] for _ in diseases_including_normal]
    conf_mat = np.zeros((disease_num_including_normal, disease_num_including_normal), dtype=int)
    
    start_time = time.time()
    corrects = 0
    corrects_codominant = 0
    total = 0
    shape = mr.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            output = mr[i][j]
            predicted = np.argmax(np.array(output))
            corrects += (predicted == i).item()
            output = torch.nn.functional.softmax(torch.Tensor(output), dim=0)
            corrects_codominant += i in utils.get_top_prob_indices(output, setting.D2_top_probs_max_num, setting.D2_top_probs_min_prob)
            total += 1
            
            conf_mat[i][predicted] += 1
            for k in range(len(diseases_including_normal)):
                disease = diseases_including_normal[i]
                roc_labels[disease].append(k == i)
                roc_probs[disease].append(output[k])
            
    acc = corrects / total
    acc_codominant = corrects_codominant / total
    time_elapsed = time.time() - start_time
    print(f"Time elapsed {time_elapsed // 60 :.0f}m {time_elapsed % 60 :.2f}s")
    print(f"acc: {corrects}/{total} = {acc :.4f}")
    print(f"acc: {corrects_codominant}/{total} = {acc_codominant :.4f}")
    
    tSNE_features = [[] for _ in range(disease_num_including_normal)]
    for i in range(len(tSNE_features)):
        #tSNE_features[i] = np.random.choice(tSNE_features[i], tSNE_point_num)
        tSNE_features[i] = mr[i][np.random.choice(mr[i].shape[0], size=tSNE_point_num, replace=False)]
        tSNE_labels[i] = [i for _ in range(tSNE_point_num)]
    tSNE_features = np.concatenate(tSNE_features)
    tSNE_labels = np.concatenate(tSNE_labels)
    
    if plt_tSNE:
        plot_tSNE(tSNE_features, tSNE_labels, abbr_diseases)
    #if plt_ROC:
    #    aucs = plot_all_ROC(roc_labels, roc_probs, diseases_including_normal)
    #if plt_conf_mat:
    #    plot_conf_mat(conf_mat, abbr_diseases)
    
    with open(os.path.join(setting.D2_folder, "000D2TORS.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([corrects, corrects_codominant, total, acc, acc_codominant])
    
    '''with open(os.path.join(setting.table_folder, "diagnosis2.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
                "Abnormity",
                "Precision",
                "Sensitivity",
                "Specificity",
                "FOne", 
                "AUC"
            ])
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
                f"{precision:.3f}",
                f"{sensitivity:.3f}",
                f"{specificity:.3f}",
                f"{f1:.3f}", 
                f"{aucs[i]:.3f}"
            ])'''