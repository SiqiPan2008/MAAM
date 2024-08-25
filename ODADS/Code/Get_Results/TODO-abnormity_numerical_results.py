import csv
import os
import numpy as np
from Utils import utils
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

def plot_complete_conf_mat(ax, complete_conf_mat, abnormities):
    mat = ax.matshow(complete_conf_mat, cmap='Blues')
    max = np.max(complete_conf_mat)
    for i in range(complete_conf_mat.shape[0]):
        for j in range(complete_conf_mat.shape[1]):
            ax.text(j, i, complete_conf_mat[i, j], ha='center', va='center', color='black' if complete_conf_mat[i, j] < max / 2 else 'white')
    plt.colorbar(mat)

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
    o_abnormities = setting.get_abnormities("OCT Abnormities")
    f_abnormities = setting.get_abnormities("Fundus Abnormities")
    o_abnormity_num = setting.get_abnormity_num("OCT Abnormities")
    f_abnormity_num = setting.get_abnormity_num("Fundus Abnormities")
    
    fig = plt.figure(figsize=(9, 8))
    axs = fig.subplots(2, 2)
    
    o_train_mr_path = os.path.join(abnormity_folder_path, o_train_mr_name + ".bin")
    o_train_mr = np.fromfile(o_train_mr_path, dtype = np.float64).reshape((o_abnormity_num, o_train_class_size, o_abnormity_num))
    o_train_conf_mat = utils.get_abnormity_complete_conf_mat(o_train_mr, o_abnormity_num)
    plot_complete_conf_mat(axs[0, 0], o_train_conf_mat, o_abnormities)
    
    o_test_mr_path = os.path.join(abnormity_folder_path, o_test_mr_name + ".bin")
    o_test_mr = np.fromfile(o_test_mr_path, dtype = np.float64).reshape((o_abnormity_num, o_test_class_size, o_abnormity_num))
    o_test_conf_mat = utils.get_abnormity_complete_conf_mat(o_test_mr, o_abnormity_num)
    plot_complete_conf_mat(axs[0, 1], o_test_conf_mat, o_abnormities)
    
    f_train_mr_path = os.path.join(abnormity_folder_path, f_train_mr_name + ".bin")
    f_train_mr = np.fromfile(f_train_mr_path, dtype = np.float64).reshape((f_abnormity_num, f_train_class_size, f_abnormity_num))
    f_train_conf_mat = utils.get_abnormity_complete_conf_mat(f_train_mr, f_abnormity_num)
    plot_complete_conf_mat(axs[1, 0], f_train_conf_mat, f_abnormities)
    
    f_test_mr_path = os.path.join(abnormity_folder_path, f_test_mr_name + ".bin")
    f_test_mr = np.fromfile(f_test_mr_path, dtype = np.float64).reshape((f_abnormity_num, f_test_class_size, f_abnormity_num))
    f_test_conf_mat = utils.get_abnormity_complete_conf_mat(f_test_mr, f_abnormity_num)
    plot_complete_conf_mat(axs[1, 1], f_test_conf_mat, f_abnormities)
    
    plt.savefig(fig_path)
    plt.show()