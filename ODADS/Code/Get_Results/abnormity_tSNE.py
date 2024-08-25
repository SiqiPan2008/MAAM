import csv
import os
import numpy as np
from Utils import utils
import sklearn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

def plot_single_tsne(ax, mr, abnormities, color_bar_on = True):
    shape = mr.shape
    features = []
    labels = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            features.append(mr[i][j])
            labels.append(abnormities[i][1])
    
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(np.array(features))
    
    scatter = ax.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap='jet', alpha=0.5)
    if color_bar_on:
        ax.colorbar(scatter)
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')

def plot_tsne():
    setting = utils.get_setting()
    fig_folder = setting.fig_folder
    fig_file = "abnormity_ROC.pdf"
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
    
    fig = plt.figure(figsize=(8, 8))
    axs = fig.subplots(2, 2)
    
    o_train_mr_path = os.path.join(abnormity_folder_path, o_train_mr_name + ".bin")
    o_train_mr = np.fromfile(o_train_mr_path, dtype = np.float64).reshape((o_abnormity_num, o_train_class_size, o_abnormity_num))
    plot_single_tsne(axs[0, 0], o_train_mr, o_abnormities)
    
    o_test_mr_path = os.path.join(abnormity_folder_path, o_test_mr_name + ".bin")
    o_test_mr = np.fromfile(o_test_mr_path, dtype = np.float64).reshape((o_abnormity_num, o_test_class_size, o_abnormity_num))
    plot_single_tsne(axs[0, 1], o_test_mr, o_abnormities)
    
    f_train_mr_path = os.path.join(abnormity_folder_path, f_train_mr_name + ".bin")
    f_train_mr = np.fromfile(f_train_mr_path, dtype = np.float64).reshape((f_abnormity_num, f_train_class_size, f_abnormity_num))
    plot_single_tsne(axs[1, 0], f_train_mr, f_abnormities)
    
    f_test_mr_path = os.path.join(abnormity_folder_path, f_test_mr_name + ".bin")
    f_test_mr = np.fromfile(f_test_mr_path, dtype = np.float64).reshape((f_abnormity_num, f_test_class_size, f_abnormity_num))
    plot_single_tsne(axs[1, 0], f_test_mr, f_abnormities)
    
    #plt.savefig(file_path)
    plt.show()