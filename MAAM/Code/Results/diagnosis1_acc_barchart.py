import csv
import os
import matplotlib.pyplot as plt
from Utils import utils
import numpy as np

# Add D2 accuracy with codominant method

def plot_acc_barchart():
    setting = utils.get_setting()
    csv_file_path = os.path.join(setting.D1_folder, "000D1TORS.csv")
    fig_folder = setting.fig_folder
    fig_file = "diagnosis1_acc_barchart.pdf"
    fig_path = os.path.join(fig_folder, fig_file)
    
    plt.rcParams["font.size"] = 9
    diseases = []
    accs = []
    with open(csv_file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            diseases.append(setting.convert_disease_to_abbr(row[0]))
            accs.append(float(row[3]))

    fig = plt.figure(figsize=(8,3))
    plt.ylim(0.0, 1.0)
    plt.axhline(y = sum(accs) / len(accs), color='gray', linestyle='--')
    
    diseases.append("")
    diseases.append("*")
    diseases.append("**")
    accs.append(0.0)
    csv_file_path = os.path.join(setting.D2_folder, "000D2TORS.csv")
    with open(csv_file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            accs.append(float(row[3]))
            accs.append(float(row[4]))
            break
    plt.bar(diseases, accs, width = 0.6)
    current_ticks = plt.xticks()[0]
    current_labels = plt.xticks()[1]
    new_ticks = [tick for tick, label in zip(current_ticks, current_labels) if tick != 12]
    new_labels = [label for tick, label in zip(current_ticks, current_labels) if tick != 12]
    plt.xticks(ticks=new_ticks, labels=new_labels)
    plt.ylim([0, 1.1])
    plt.yticks(ticks=[0,0.2,0.4,0.6,0.8,1.0])
    plt.text(-0.9, 0.79, "Stage D1 Average", color = "gray")
    plt.text(4.75, -0.25, "Stage D1", fontsize = 12)
    plt.text(12.75, -0.25, "Stage D2", fontsize = 12)
    
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.show()