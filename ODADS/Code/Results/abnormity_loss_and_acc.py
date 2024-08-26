import csv
import os
import numpy as np
from Utils import utils
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
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
    best_epoch = int(best_epoch)
    best_acc = float(best_acc)
    
    epoch_train = [int(row[0]) + starting_epoch for row in train_rs]
    valid_acc = [float(row[1]) for row in train_rs]
    train_acc = [float(row[2]) for row in train_rs]
    valid_losses = [float(row[3]) for row in train_rs]
    train_losses = [float(row[4]) for row in train_rs]
    epoch_test = [int(row[0]) + starting_epoch for row in test_rs]
    test_acc = [float(row[3]) for row in test_rs]
    
    return {
        "epoch train": epoch_train,
        "valid acc": valid_acc,
        "train acc": train_acc,
        "valid losses": valid_losses,
        "train losses": train_losses,
        "epoch test": epoch_test,
        "test acc": test_acc,
        "best epoch": best_epoch,
        "best acc": best_acc
    }

def plot_loss_and_acc_alt(): # (2x1) x (2x2) x (2x1)
    setting = utils.get_setting()
    fig_folder = setting.fig_folder
    fig_file = "abnormity_loss_and_acc.pdf"
    fig_path = os.path.join(fig_folder, fig_file)
    
    fig = plt.figure(figsize=(8, 12))
    grid = gridspec.GridSpec(2, 1, wspace = 0, hspace = 0.3)
    plt.axis(False)

    model_list = [
        ("AF", 0, "152"), 
        ("AO", 1, "050")
    ]

    net_list = [
        ("152", 0, 0), 
        ("050", 0, 1),
        ("018", 1, 0),
        ("016", 1, 1)
    ]
    
    for type, pos, best_net in model_list:
        sub_ax = fig.add_subplot(grid[pos, 0])
        sub_ax.set_title("Fundus" if type == "AF" else "OCT")
        sub_ax.set_axis_off()
        sub_grid = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec = grid[pos, 0], wspace = 0.3, hspace = 0.3)
        
        for net_name, pos_x, pos_y in net_list:
            subsub_ax = fig.add_subplot(sub_grid[pos_x, pos_y])
            subsub_ax.set_title(net_name)
            subsub_ax.set_axis_off()
            subsub_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec = sub_grid[pos_x, pos_y], wspace = 0, hspace = 0.1)
            
            rst = get_loss_and_acc(net_name, type) # ReSults from Transferred learning
            rsf = get_loss_and_acc(net_name, type, starting_epoch = rst["best epoch"]) # ReSults from Finetuning
            continuation_epoch = rst["best epoch"]
            rsf["epoch train"].insert(0, continuation_epoch)
            rsf["train acc"].insert(0, rst["train acc"][continuation_epoch - 1])
            rsf["train losses"].insert(0, rst["train losses"][continuation_epoch - 1])
            rsf["valid acc"].insert(0, rst["valid acc"][continuation_epoch - 1])
            rsf["valid losses"].insert(0, rst["valid losses"][continuation_epoch - 1])
            rsf["epoch test"].insert(0, continuation_epoch)
            rsf["test acc"].insert(0, rst["test acc"][rst["epoch test"].index(continuation_epoch)])
            
            acc_ax = fig.add_subplot(subsub_grid[0, 0])
            acc_ax.plot(rst["epoch train"], rst["train acc"], color='blue')
            acc_ax.plot(rsf["epoch train"], rsf["train acc"], color='blue')
            acc_ax.plot(rst["epoch train"], rst["valid acc"], color='red')
            acc_ax.plot(rsf["epoch train"], rsf["valid acc"], color='red')
            acc_ax.plot(rst["epoch test"], rst["test acc"], color='green')
            acc_ax.plot(rsf["epoch test"], rsf["test acc"], color='green')
            
            loss_ax = fig.add_subplot(subsub_grid[1, 0])
            loss_ax.plot(rst["epoch train"], rst["train losses"], color='blue')
            loss_ax.plot(rsf["epoch train"], rsf["train losses"], color='blue')
            loss_ax.plot(rst["epoch train"], rst["valid losses"], color='red')
            loss_ax.plot(rsf["epoch train"], rsf["valid losses"], color='red')
    
    plt.show()
    plt.save(fig_path)
    
    
    
    
    
    
    
    
def plot_loss_and_acc(): # 2 x (4x1) x (2x2)
    setting = utils.get_setting()
    fig_folder = setting.fig_folder
    fig_file = "abnormity_loss_and_acc.pdf"
    fig_path = os.path.join(fig_folder, fig_file)
    
    fig = plt.figure(figsize=(8, 13))
    grid = gridspec.GridSpec(4, 1, wspace = 0, hspace = 0.2)
    plt.axis(False)
    
    type = "AF" # "AO"
    best_net = "152" # "050"

    net_list = [
        ("152", 0), 
        ("050", 1),
        ("018", 2),
        ("016", 3)
    ]
    
    plt.title("Fundus" if type == "AF" else "OCT")
    
    for net_name, pos in net_list:
        sub_ax = fig.add_subplot(grid[pos, 0])
        sub_ax.set_title(net_name)
        sub_ax.set_axis_off()
        sub_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec = grid[pos, 0], wspace = 0, hspace = 0.05)
        
        rst = get_loss_and_acc(net_name, type) # ReSults from Transferred learning
        rsf = get_loss_and_acc(net_name, type, starting_epoch = rst["best epoch"]) # ReSults from Finetuning
        continuation_epoch = rst["best epoch"]
        rsf["epoch train"].insert(0, continuation_epoch)
        rsf["train acc"].insert(0, rst["train acc"][continuation_epoch - 1])
        rsf["train losses"].insert(0, rst["train losses"][continuation_epoch - 1])
        rsf["valid acc"].insert(0, rst["valid acc"][continuation_epoch - 1])
        rsf["valid losses"].insert(0, rst["valid losses"][continuation_epoch - 1])
        rsf["epoch test"].insert(0, continuation_epoch)
        rsf["test acc"].insert(0, rst["test acc"][rst["epoch test"].index(continuation_epoch)])
        
        acc_ax = fig.add_subplot(sub_grid[0, 0])
        acc_ax.plot(rst["epoch train"], rst["train acc"], color='blue')
        acc_ax.plot(rsf["epoch train"], rsf["train acc"], color='blue')
        acc_ax.plot(rst["epoch train"], rst["valid acc"], color='red')
        acc_ax.plot(rsf["epoch train"], rsf["valid acc"], color='red')
        acc_ax.plot(rst["epoch test"], rst["test acc"], color='green')
        acc_ax.plot(rsf["epoch test"], rsf["test acc"], color='green')
        
        loss_ax = fig.add_subplot(sub_grid[1, 0])
        loss_ax.plot(rst["epoch train"], rst["train losses"], color='blue')
        loss_ax.plot(rsf["epoch train"], rsf["train losses"], color='blue')
        loss_ax.plot(rst["epoch train"], rst["valid losses"], color='red')
        loss_ax.plot(rsf["epoch train"], rsf["valid losses"], color='red')
    
    plt.savefig(fig_path)
    plt.show()