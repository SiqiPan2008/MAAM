import csv
import os
import numpy as np
from matplotlib.lines import Line2D
from Utils import utils
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_single_roc(type, abnormity, draw_roc = True, ax = None, row = 0, col = 0):
    colors = {"TR": "#FF0000", "TO": "#00BB00"}
    linewidths = {"TR": 1.2, "TO": 1.2}
    
    setting = utils.get_setting()
    abbr = setting.convert_abnormity_to_abbr(abnormity, type)
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
            ax.plot(fpr, tpr, color = colors[source], linewidth = linewidths[source])
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.0])
    if draw_roc:
        ax.plot([0, 1], [0, 1], color = "gray", linewidth = 0.6)
        ax.text(0.95, 0.05, abbr, verticalalignment='bottom', horizontalalignment='right', fontsize=9)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.xaxis.set_ticks_position('top')
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.xaxis.set_tick_params(labelbottom=False, labeltop=True)
        if row == 0 or row == 3:
            ax.set_xticklabels(["0", "", "0.5", "", "1"])
        else:
            ax.set_xticks([])
        if col == 0:
            ax.set_yticklabels(["0", "", "0.5", "", "1"])
        else:
            ax.set_yticks([])
    return aucs

def plot_roc():
    setting = utils.get_setting()
    fig_folder = setting.fig_folder
    fig_file = "abnormity_ROC.pdf"
    fig_path = os.path.join(fig_folder, fig_file)
    o_abnormities = setting.get_abnormities("OCT Abnormities")
    f_abnormities = setting.get_abnormities("Fundus Abnormities")
    
    plt.rcParams['axes.titlesize'] = 9
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    
    fig = plt.figure(figsize=(8, 6))
    grid = gridspec.GridSpec(5, 6, height_ratios=[1, 1, 0.45, 1, 1], hspace=0.125, wspace=0.1)
    
    row = 0
    col = 0
    for abnormity in o_abnormities:
        ax = fig.add_subplot(grid[row, col])
        plot_single_roc("OCT", abnormity[1], ax = ax, row = row, col = col)
        if col == 5:
            col = 0
            row += 1
        else:
            col += 1
    
    row += 1
    for abnormity in f_abnormities:
        ax = fig.add_subplot(grid[row, col])
        plot_single_roc("Fundus", abnormity[1], ax = ax, row = row, col = col)
        if row == 4 and col == 3:
            break
        elif col == 5:
            col = 0
            row += 1
        else:
            col += 1
    
    fig.text(0.5, 0.925, "OCT", verticalalignment='bottom', horizontalalignment='center', fontsize=12)
    fig.text(0.5, 0.485, "Fundus", verticalalignment='bottom', horizontalalignment='center', fontsize=12)
    colors = {"TR": "#FF0000", "TO": "#00BB00"}
    legend_lines = [
        Line2D([0], [0], color=colors["TR"], linewidth = 1.2, linestyle='-', label='ROC for training'),
        Line2D([0], [0], color=colors["TO"], linewidth = 1.2,linestyle='-', label='ROC for testing'),
    ]
    fig.legend(handles=legend_lines, ncols = 2, loc = "lower right", bbox_to_anchor = (0.91,0.1))
    fig.text(0.898, 0.17, "Horizontal axes: specificity\nVertical axes: sensitivity", fontsize=9, 
        verticalalignment='bottom', horizontalalignment='right', bbox=dict(facecolor='white', edgecolor='#D1D1D1', boxstyle='round,pad=0.25'))
    plt.savefig(fig_path)
    plt.show()