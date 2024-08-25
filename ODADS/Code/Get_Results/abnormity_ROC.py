import csv
import os
import numpy as np
from Utils import utils
import sklearn
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_single_roc(type, abnormity, draw_roc = True, ax = None):
    colors = {"TR": "red", "TO": "green"}
    
    setting = utils.get_setting()
    abnormities = setting.get_abnormities("OCT Abnormities") if type == "OCT" else setting.get_abnormities("Fundus Abnormities")
    abnormity_index = abnormities.index((type, abnormity))
    abnormity_folder_path = setting.A_folder
    best_net = setting.best_o_net if type == "OCT" else setting.best_f_net
    class_sizes = {
        "TR": setting.O_train_class_size if type == "OCT" else setting.F_train_class_size,
        "TO": setting.O_test_class_size if type == "OCT" else setting.F_test_class_size
    }
    abnormity_num = setting.get_abnormity_num("OCT Abnormities") if type == "OCT" else setting.get_abnormity_num("Fundus Abnormities")
    
    aucs = {"TR": 0.0, "TO": 0.0}
    for source in ["TR", "TO"]:
        if source == "TR":
            mr_name = setting.get_o_trmr_name(best_net) if type == "OCT" else setting.get_f_trmr_name(best_net)
        elif source == "TO":
            mr_name = setting.get_o_tomr_name(best_net) if type == "OCT" else setting.get_f_tomr_name(best_net)
        mr_path = os.path.join(abnormity_folder_path, mr_name + ".bin")
        mr = np.fromfile(mr_path, dtype = np.float64).reshape((abnormity_num, class_sizes[source], abnormity_num))

        labels = np.zeros((abnormity_num * class_sizes[source]))
        probs = np.zeros((abnormity_num * class_sizes[source]))
        for i in range(abnormity_num):
            for j in range(class_sizes[source]):
                index = i * class_sizes[source] + j
                if i == abnormity_index:
                    labels[index] = 1
                probs[index] = mr[i][j][abnormity_index]
                
        fpr, tpr, _ = roc_curve(labels, probs)
        aucs[source] = auc(fpr, tpr)
        if draw_roc:
            ax.plot(fpr, tpr, color = colors[source])
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.0])
    if draw_roc:
        ax.plot([0, 1], [0, 1], color = "black")
    return aucs

def plot_roc():
    setting = utils.get_setting()
    fig_folder = setting.fig_folder
    fig_file = "abnormity_ROC.pdf"
    fig_path = os.path.join(fig_folder, fig_file)
    o_abnormities = setting.get_abnormities("OCT Abnormities")
    f_abnormities = setting.get_abnormities("Fundus Abnormities")
    
    fig = plt.figure(figsize=(8, 12))
    axs = fig.subplots(6, 4)
    
    row = 0
    col = 0
    for abnormity in o_abnormities:
        plot_single_roc("OCT", abnormity[1], ax = axs[row, col])
        if col == 3:
            col = 0
            row += 1
        else:
            col += 1
    
    for i in range(1, 4):
        axs[3, i].set_axis_off()
    for abnormity in f_abnormities:
        plot_single_roc("Fundus", abnormity[1], ax = axs[row, col])
        if row == 3:
            row += 1
        elif col == 3:
            col = 0
            row += 1
        else:
            col += 1
    
    plt.show()