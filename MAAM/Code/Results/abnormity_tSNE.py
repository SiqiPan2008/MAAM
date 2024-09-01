import csv
import os
import numpy as np
from Utils import utils
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

def plot_single_tsne(ax, mr, abnormities, tSNE_point_num):
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
    for i, disease in enumerate(diseases):
        handle = Line2D([], [], color=plt.get_cmap('jet')(i / len(diseases)), marker='o', linestyle='None', markersize=s, label=disease)
        handles.append(handle)
    plt.legend(handles=handles, bbox_to_anchor=(1, 0.5), loc='center left')
    ax.set_xticks([])
    ax.set_yticks([])

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
    o_abnormities = setting.get_abnormities("OCT Abnormities")
    f_abnormities = setting.get_abnormities("Fundus Abnormities")
    o_abnormity_num = setting.get_abnormity_num("OCT Abnormities")
    f_abnormity_num = setting.get_abnormity_num("Fundus Abnormities")
    
    fig = plt.figure(figsize=(9, 8))
    axs = fig.subplots(2, 2)
    
    o_train_mr_path = os.path.join(abnormity_folder_path, o_train_mr_name + ".bin")
    o_train_mr = np.fromfile(o_train_mr_path, dtype = np.float64).reshape((o_abnormity_num, o_train_class_size, o_abnormity_num))
    plot_single_tsne(axs[0, 0], o_train_mr, o_abnormities, tSNE_point_num)
    
    o_test_mr_path = os.path.join(abnormity_folder_path, o_test_mr_name + ".bin")
    o_test_mr = np.fromfile(o_test_mr_path, dtype = np.float64).reshape((o_abnormity_num, o_test_class_size, o_abnormity_num))
    plot_single_tsne(axs[0, 1], o_test_mr, o_abnormities, tSNE_point_num)
    
    f_train_mr_path = os.path.join(abnormity_folder_path, f_train_mr_name + ".bin")
    f_train_mr = np.fromfile(f_train_mr_path, dtype = np.float64).reshape((f_abnormity_num, f_train_class_size, f_abnormity_num))
    plot_single_tsne(axs[1, 0], f_train_mr, f_abnormities, tSNE_point_num)
    
    f_test_mr_path = os.path.join(abnormity_folder_path, f_test_mr_name + ".bin")
    f_test_mr = np.fromfile(f_test_mr_path, dtype = np.float64).reshape((f_abnormity_num, f_test_class_size, f_abnormity_num))
    plot_single_tsne(axs[1, 1], f_test_mr, f_abnormities, tSNE_point_num)
    
    plt.savefig(fig_path)
    plt.show()