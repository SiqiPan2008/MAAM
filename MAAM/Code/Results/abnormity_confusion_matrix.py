import csv
import os
import numpy as np
from Utils import utils
from sklearn.metrics import roc_curve, auc
from Results import abnormity_ROC
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


def get_single_conf_mat(complete_conf_mat, index):
    conf_mat = np.zeros((2, 2), dtype=int)
    num = complete_conf_mat.shape[0]
    for i in range(num):
        for j in range(num):
            conf_mat[1 if i == index else 0, 1 if j == index else 0] += complete_conf_mat[i, j]
    return conf_mat

def get_complete_conf_mat(mr, abnormity_num):
    conf_mat = np.zeros((abnormity_num, abnormity_num), dtype=int)
    shape = mr.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            probs = mr[i][j]
            conf_mat[i][np.argmax(probs)] += 1
    return conf_mat

def plot_complete_conf_mat(ax, complete_conf_mat, abnormities, fontsize, title):
    mat = ax.matshow(complete_conf_mat, cmap='Blues')
    max = np.max(complete_conf_mat)
    for i in range(complete_conf_mat.shape[0]):
        for j in range(complete_conf_mat.shape[1]):
            ax.text(j, i, complete_conf_mat[i, j], ha='center', va='center', color='black' if complete_conf_mat[i, j] < max / 2 else 'white', fontsize = fontsize)
    ax.set_xticks(np.arange(len(abnormities)))
    ax.set_yticks(np.arange(len(abnormities)))
    ax.set_xticklabels(abnormities, rotation=-90)
    ax.set_yticklabels(abnormities)
    ax.tick_params(axis='x', which='both', bottom=False)
    ax.set_title(title, fontsize = 12)

def calc_and_save_numerical_results(csv_filename, type, source, conf_mat):
    setting = utils.get_setting()
    table_folder = setting.table_folder
    table_path = os.path.join(table_folder, csv_filename + ".csv")
    abnormities = setting.get_abnormities("OCT Abnormities") if type == "OCT" else setting.get_abnormities("Fundus Abnormities")
    abbr_abnormities = setting.get_abbr_abnormities("OCT Abnormities") if type == "OCT" else setting.get_abbr_abnormities("Fundus Abnormities")
    with open(table_path, 'w', newline = "") as file:
        writer = csv.writer(file)
        writer.writerow([
                "Abnormity",
                "Precision",
                "Sensitivity",
                "Specificity",
                "FOne", 
                "AUC"
            ])
        for i in range(len(abnormities)):
            auc = abnormity_ROC.plot_single_roc(type, abnormities[i][1], draw_roc = False)[source]
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
                abbr_abnormities[i][1],
                f"{precision:.3f}",
                f"{sensitivity:.3f}",
                f"{specificity:.3f}",
                f"{f1:.3f}", 
                f"{auc:.3f}"
            ])

def plot_all_conf_mat():
    setting = utils.get_setting()
    fig_folder = setting.fig_folder
    fig_file = "abnormity_confusion_matrix.pdf"
    abnormity_folder_path = setting.A_folder
    fig_path = os.path.join(fig_folder, fig_file)
    o_test_class_size = setting.O_test_class_size
    f_test_class_size = setting.F_test_class_size
    o_train_class_size = setting.O_train_class_size
    f_train_class_size = setting.F_train_class_size
    o_test_mr_name = setting.get_o_tomr_name(setting.best_o_net)
    f_test_mr_name = setting.get_f_tomr_name(setting.best_f_net)
    o_train_mr_name = setting.get_o_trmr_name(setting.best_o_net)
    f_train_mr_name = setting.get_f_trmr_name(setting.best_f_net)
    o_abbr_abnormities = setting.get_abbr_abnormities("OCT Abnormities")
    o_abbr_abnormities = [abnormity[1] for abnormity in o_abbr_abnormities]
    f_abbr_abnormities = setting.get_abbr_abnormities("Fundus Abnormities")
    f_abbr_abnormities = [abnormity[1] for abnormity in f_abbr_abnormities]
    o_abnormity_num = setting.get_abnormity_num("OCT Abnormities")
    f_abnormity_num = setting.get_abnormity_num("Fundus Abnormities")
    
    plt.rcParams['font.size'] = 9
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    
    fig = plt.figure(figsize=(8, 8))
    axs = fig.subplots(2, 2)
    fig.subplots_adjust(hspace = 0.35, wspace = 0.2)
    
    o_train_mr_path = os.path.join(abnormity_folder_path, o_train_mr_name + ".bin")
    o_train_mr = np.fromfile(o_train_mr_path, dtype = np.float64).reshape((o_abnormity_num, o_train_class_size, o_abnormity_num))
    o_train_conf_mat = get_complete_conf_mat(o_train_mr, o_abnormity_num)
    plot_complete_conf_mat(axs[0, 0], o_train_conf_mat, o_abbr_abnormities, 6, "OCT Model Training")
    calc_and_save_numerical_results("abnormity_o_train", "OCT", "TR", o_train_conf_mat)
    
    o_test_mr_path = os.path.join(abnormity_folder_path, o_test_mr_name + ".bin")
    o_test_mr = np.fromfile(o_test_mr_path, dtype = np.float64).reshape((o_abnormity_num, o_test_class_size, o_abnormity_num))
    o_test_conf_mat = get_complete_conf_mat(o_test_mr, o_abnormity_num)
    plot_complete_conf_mat(axs[0, 1], o_test_conf_mat, o_abbr_abnormities, 7, "OCT Model Testing")
    calc_and_save_numerical_results("abnormity_o_test", "OCT", "TO", o_test_conf_mat)
    
    f_train_mr_path = os.path.join(abnormity_folder_path, f_train_mr_name + ".bin")
    f_train_mr = np.fromfile(f_train_mr_path, dtype = np.float64).reshape((f_abnormity_num, f_train_class_size, f_abnormity_num))
    f_train_conf_mat = get_complete_conf_mat(f_train_mr, f_abnormity_num)
    plot_complete_conf_mat(axs[1, 0], f_train_conf_mat, f_abbr_abnormities, 8, "Fundus Model Training")
    calc_and_save_numerical_results("abnormity_f_train", "Fundus", "TR", f_train_conf_mat)
    
    f_test_mr_path = os.path.join(abnormity_folder_path, f_test_mr_name + ".bin")
    f_test_mr = np.fromfile(f_test_mr_path, dtype = np.float64).reshape((f_abnormity_num, f_test_class_size, f_abnormity_num))
    f_test_conf_mat = get_complete_conf_mat(f_test_mr, f_abnormity_num)
    plot_complete_conf_mat(axs[1, 1], f_test_conf_mat, f_abbr_abnormities, 9, "Fundus Model Testing")
    calc_and_save_numerical_results("abnormity_f_test", "Fundus", "TO", f_test_conf_mat)
    
    plt.savefig(fig_path, bbox_inches = "tight")
    plt.show()