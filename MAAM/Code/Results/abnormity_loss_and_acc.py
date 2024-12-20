import csv
import os
import numpy as np
from Utils import utils
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

def read_csv(path):
    result = []
    best_epoch = 0
    best_acc = 0
    with open(path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) > 2:
                result.append(row)
            elif len(row) == 2:
                best_epoch, best_acc = row
    if best_acc != 0:
        return result, best_epoch, best_acc
    else:
        return result

def get_loss_and_acc(net_name, type, starting_epoch = 0):
    setting = utils.get_setting()
    A_folder = setting.A_folder
    name = net_name + type + \
        ("---- - T.csv" if starting_epoch == 0 else "---- - F.csv")
    train_rs_file = os.path.join(A_folder, setting.get_training_rs_name(name))
    test_rs_file = os.path.join(A_folder, setting.get_testing_rs_name(name))
    train_rs = read_csv(train_rs_file)
    test_rs, best_epoch, best_acc = read_csv(test_rs_file)
    best_epoch = int(best_epoch) + starting_epoch
    best_acc = float(best_acc)
    
    epoch_train = [int(row[0]) + starting_epoch for row in train_rs]
    valid_acc = [float(row[1]) for row in train_rs]
    train_acc = [float(row[2]) for row in train_rs]
    valid_losses = [float(row[3]) for row in train_rs]
    train_losses = [float(row[4]) for row in train_rs]
    epoch_test = [int(row[0]) + starting_epoch for row in test_rs]
    test_acc = [float(row[3]) for row in test_rs]
    combined_acc = [(test_acc[i] + 2 * valid_acc[epoch_train.index(epoch_test[i])]) / 3 for i in range(len(epoch_test))]
    
    return {
        "epoch train": epoch_train,
        "valid acc": valid_acc,
        "train acc": train_acc,
        "valid losses": valid_losses,
        "train losses": train_losses,
        "epoch test": epoch_test,
        "test acc": test_acc,
        "best epoch": best_epoch,
        "best acc": best_acc,
        "combined acc": combined_acc
    }

def plot_loss_and_acc():
    setting = utils.get_setting()
    fig_folder = setting.fig_folder
    net_list = [
        ("152", "ResNet152", (0, 0)), 
        ("050", "ResNet50", (0, 1)),
        ("018", "ResNet18", (1, 0)),
        ("016", "VGG16", (1, 1))
    ]
    
    plt.rcParams['axes.titlesize'] = 9
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['lines.linewidth'] = 1
    
    for type in ["AF", "AO"]:
        best_net = setting.best_f_net if type == "AF" else setting.best_o_net
        fig_file = "abnormity_Fundus_loss_and_acc.pdf" if type == "AF" else "abnormity_OCT_loss_and_acc.pdf"
        fig_path = os.path.join(fig_folder, fig_file)
        
        fig = plt.figure(figsize=(8, 5) if type == "AO" else (8, 5.6))
        grid = gridspec.GridSpec(2, 2, wspace = 0.05, hspace = 0.25)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9 if type == "AO" else 0.95, bottom=0.1 if type == "AO" else 0.2357142857142857)
        plt.axis(False)
        plt.title("Fundus" if type == "AF" else "OCT", fontsize = 12)
        
        for net_name, net_name_title, pos in net_list:
            sub_ax = fig.add_subplot(grid[pos])
            if pos[0] == 1:
                sub_ax.set_title(net_name_title, pad = 3.5)
            else:
                sub_ax.text(0.5, -0.03, net_name_title,ha='center', va='top', fontsize=9)
            sub_ax.set_axis_off()
            sub_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec = grid[pos], wspace = 0, hspace = 0.025)
            
            rst = get_loss_and_acc(net_name, type) # ReSults from Transferred learning
            rsf = get_loss_and_acc(net_name, type, starting_epoch = rst["best epoch"]) # ReSults from Finetuning
            continuation_epoch = rst["best epoch"]
            best_epoch = rst["best epoch"] if rst["best acc"] > rsf["best acc"] else rsf["best epoch"]
            rsf["epoch train"].insert(0, continuation_epoch)
            rsf["train acc"].insert(0, rst["train acc"][continuation_epoch - 1])
            rsf["train losses"].insert(0, rst["train losses"][continuation_epoch - 1])
            rsf["valid acc"].insert(0, rst["valid acc"][continuation_epoch - 1])
            rsf["valid losses"].insert(0, rst["valid losses"][continuation_epoch - 1])
            rsf["epoch test"].insert(0, continuation_epoch)
            rsf["test acc"].insert(0, rst["test acc"][rst["epoch test"].index(continuation_epoch)])
            rsf["combined acc"].insert(0, rst["combined acc"][rst["epoch test"].index(continuation_epoch)])
            
            c = ['#0000DD', '#9900EE', '#DD0000', '#DDAA00', '#00DD00', '#00DDEE']
            
            acc_ax = fig.add_subplot(sub_grid[0, 0])
            acc_ax.plot(rst["epoch train"], rst["train acc"], color=c[0])
            acc_ax.plot(rsf["epoch train"], rsf["train acc"], color=c[1])
            acc_ax.plot(rst["epoch train"], rst["valid acc"], color=c[2])
            acc_ax.plot(rsf["epoch train"], rsf["valid acc"], color=c[3])
            acc_ax.plot(rst["epoch test"], rst["combined acc"], color=c[4])
            acc_ax.plot(rsf["epoch test"], rsf["combined acc"], color=c[5])
            acc_ax.axvline(x=continuation_epoch, linewidth = 0.7, color='gray', linestyle='--')
            acc_ax.axvline(x=best_epoch, linewidth = 0.5, color='gray', linestyle='-')
            
            loss_ax = fig.add_subplot(sub_grid[1, 0])
            loss_ax.plot(rst["epoch train"], rst["train losses"], color=c[0])
            loss_ax.plot(rsf["epoch train"], rsf["train losses"], color=c[1])
            loss_ax.plot(rst["epoch train"], rst["valid losses"], color=c[2])
            loss_ax.plot(rsf["epoch train"], rsf["valid losses"], color=c[3])
            loss_ax.axvline(x=continuation_epoch, linewidth = 0.7, color='gray', linestyle='--')
            loss_ax.axvline(x=best_epoch, linewidth = 0.5, color='gray', linestyle='-')

            
            acc_ax.xaxis.set_ticks([])
            if pos[1] == 1:
                acc_ax.yaxis.tick_right()
                acc_ax.yaxis.set_label_position('right')
                loss_ax.yaxis.tick_right()
                loss_ax.yaxis.set_label_position('right')
            else:
                acc_ax.set_ylabel("Accuracy")
                loss_ax.set_ylabel("Loss")
            if pos[0] == 0:
                loss_ax.xaxis.set_ticks([])
            else:
                loss_ax.set_xlabel("Epoch")
        
        if type == "AF" and pos == (1, 1):
            
            legend_lines = [
                Line2D([0], [0], color=c[0], linestyle='-', label='Transfer learning training accuracy / loss'),
                Line2D([0], [0], color=c[2], linestyle='-', label='Transfer learning validation accuracy / loss'),
                Line2D([0], [0], color=c[4], linestyle='-', label='Transfer learning overall accuracy'),
                Line2D([0], [0], color="gray", linestyle='--', linewidth = 0.7, label='Finetuning startpoint'),
                Line2D([0], [0], color=c[1], linestyle='-', label='Finetuning training accuracy / loss'),
                Line2D([0], [0], color=c[3], linestyle='-', label='Finetuning validation accuracy / loss'),
                Line2D([0], [0], color=c[5], linestyle='-', label='Finetuning overall accuracy'),
                Line2D([0], [0], color="gray", linestyle='-', linewidth = 0.5, label='Best model')
            ]

            fig.legend(handles=legend_lines, ncols = 2, loc = "upper center", bbox_to_anchor = (0.5,0.16))
            
        plt.savefig(fig_path)
        plt.show()