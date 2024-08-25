import csv
import os
import numpy as np
from Utils import utils
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def read_csv(path):
    setting = utils.get_setting()
    best_o_net = setting.best_o_net
    best_f_net = setting.best_f_net
    abnormity_folder_path = setting.A_folder
    o_class_size = setting.O_train_class_size
    f_class_size = setting.F_train_class_size
    o_abnormity_num = setting.get_abnormity_num("OCT Abnormities")
    f_abnormity_num = setting.get_abnormity_num("Fundus Abnormities")
    
    for source in ["TR", "TO"]:
        if source == "TR":
            o_mr_name = setting.get_o_trmr_name(best_o_net)
            f_mr_name = setting.get_f_trmr_name(best_f_net)
        elif source == "TO":
            o_mr_name = setting.get_o_tomr_name(best_o_net)
            f_mr_name = setting.get_f_tomr_name(best_f_net)
        o_mr_path = os.path.join(abnormity_folder_path, o_mr_name + ".bin")
        o_mr = np.fromfile(o_mr_path, dtype = np.float64).reshape((o_abnormity_num, o_class_size, o_abnormity_num))
        f_mr_path = os.path.join(abnormity_folder_path, f_mr_name + ".bin")
        f_mr = np.fromfile(f_mr_path, dtype = np.float64).reshape((f_abnormity_num, f_class_size, f_abnormity_num))
        
        

def graph(): # 2 x (4x1) x (2x2)
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