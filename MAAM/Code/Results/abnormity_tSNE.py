import csv
import os
import numpy as np
from Utils import utils
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_single_tsne(ax, mr, abnormities, tSNE_point_num, add_legends, title):
    shape = mr.shape
    features = []
    labels = []
    for i in range(shape[0]):
        class_features = []
        for j in range(shape[1]):
            class_features.append(mr[i][j])
        class_features = random.sample(class_features, tSNE_point_num)
        class_labels = [i for _ in range(tSNE_point_num)]
        features.append(np.array(class_features))
        labels.append(np.array(class_labels))
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    
    tsne = TSNE(n_components=2)
    features_tsne = tsne.fit_transform(np.array(features))
    
    s=2
    scatter = ax.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap='jet', marker='o', s=s)
    if add_legends:
        handles = []
        for i, disease in enumerate(abnormities):
            handle = Line2D([], [], color=plt.get_cmap('jet')(i / len(abnormities)), marker='o', linestyle='None', markersize=s, label=disease)
            handles.append(handle)
        ax.legend(handles=handles, bbox_to_anchor=(1.025, 0.5), loc='center left')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, pad = 10)

def plot_tsne():
    setting = utils.get_setting()
    fig_folder = setting.fig_folder
    fig_file = "abnormity_tSNE.pdf"
    tSNE_point_num = setting.tSNE_point_num
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
    o_abnormities = setting.get_abbr_abnormities("OCT Abnormities")
    o_abnormities = [abnormity[1] for abnormity in o_abnormities]
    f_abnormities = setting.get_abbr_abnormities("Fundus Abnormities")
    f_abnormities = [abnormity[1] for abnormity in f_abnormities]
    o_abnormity_num = setting.get_abnormity_num("OCT Abnormities")
    f_abnormity_num = setting.get_abnormity_num("Fundus Abnormities")
    
    fig = plt.figure(figsize=(8, 7.5))
    axs = fig.subplots(2, 2)
    fig.subplots_adjust(wspace = 0.2, hspace = 0.7)
    
    o_train_mr_path = os.path.join(abnormity_folder_path, o_train_mr_name + ".bin")
    o_train_mr = np.fromfile(o_train_mr_path, dtype = np.float64).reshape((o_abnormity_num, o_train_class_size, o_abnormity_num))
    plot_single_tsne(axs[0, 0], o_train_mr, o_abnormities, tSNE_point_num, False, "OCT Model Training")
    
    o_test_mr_path = os.path.join(abnormity_folder_path, o_test_mr_name + ".bin")
    o_test_mr = np.fromfile(o_test_mr_path, dtype = np.float64).reshape((o_abnormity_num, o_test_class_size, o_abnormity_num))
    plot_single_tsne(axs[0, 1], o_test_mr, o_abnormities, tSNE_point_num, True, "OCT Model Testing")
    
    f_train_mr_path = os.path.join(abnormity_folder_path, f_train_mr_name + ".bin")
    f_train_mr = np.fromfile(f_train_mr_path, dtype = np.float64).reshape((f_abnormity_num, f_train_class_size, f_abnormity_num))
    plot_single_tsne(axs[1, 0], f_train_mr, f_abnormities, tSNE_point_num, False, "Fundus Model Training")
    
    f_test_mr_path = os.path.join(abnormity_folder_path, f_test_mr_name + ".bin")
    f_test_mr = np.fromfile(f_test_mr_path, dtype = np.float64).reshape((f_abnormity_num, f_test_class_size, f_abnormity_num))
    plot_single_tsne(axs[1, 1], f_test_mr, f_abnormities, tSNE_point_num, True, "Fundus Model Testing")
    
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.show()