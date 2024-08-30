import csv
import os
import matplotlib.pyplot as plt
from Utils import utils

# Add D2 accuracy with codominant method

def plot_acc_barchart():
    setting = utils.get_setting()
    csv_file_path = os.path.join(setting.D1_folder, "000D1TORS.csv")
    fig_folder = setting.fig_folder
    fig_file = "diagnosis1_acc_barchart.pdf"
    fig_path = os.path.join(fig_folder, fig_file)
    
    diseases = []
    accs = []
    with open(csv_file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            diseases.append(row[0])
            accs.append(float(row[3]))

    
    plt.ylim(0.0, 1.0)
    plt.axhline(y = sum(accs) / len(accs), color='gray', linestyle='--')
    
    diseases.append("")
    diseases.append("D2")
    accs.append(0.0)
    csv_file_path = os.path.join(setting.D2_folder, "000D2TORS.csv")
    with open(csv_file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            accs.append(float(row[2]))
            break
    plt.bar(diseases, accs)
    
    plt.savefig(fig_path)
    plt.show()