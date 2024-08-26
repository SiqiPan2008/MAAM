import csv
import os
import numpy as np
from Utils import utils
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import time
import torch
import matplotlib.gridspec as gridspec
from Diagnosis_Model import diagnosis_model

def get_conf_mat(device, disease):
    setting = utils.get_setting()
    name = "000D1---- - " + disease
    batch_size = setting.test_batch_size
    class_size = setting.D1_test_class_size
    abnormity_folder_path = setting.A_folder
    o_class_size = setting.O_test_class_size
    f_class_size = setting.F_test_class_size
    wt_name = setting.get_d_wt_file_name(name)
    o_mr_name = setting.get_o_tomr_name(setting.best_o_net)
    f_mr_name = setting.get_f_tomr_name(setting.best_f_net)
    folder_path = setting.get_folder_path(name)
    o_abnormity_num = setting.get_abnormity_num("OCT Abnormities")
    f_abnormity_num = setting.get_abnormity_num("Fundus Abnormities")
    all_abnormity_num = setting.get_abnormity_num("All Abnormities")
    label_levels = setting.get_disease_abnormity_num(disease, "All Abnormities") + 1
    
    o_mr_path = os.path.join(abnormity_folder_path, o_mr_name + ".bin")
    o_mr = np.fromfile(o_mr_path, dtype = np.float64).reshape((o_abnormity_num, o_class_size, o_abnormity_num))
    f_mr_path = os.path.join(abnormity_folder_path, f_mr_name + ".bin")
    f_mr = np.fromfile(f_mr_path, dtype = np.float64).reshape((f_abnormity_num, f_class_size, f_abnormity_num))
    
    test_data = torch.zeros(label_levels * class_size, all_abnormity_num)
    test_label = torch.zeros(label_levels * class_size, dtype=torch.long)
    
    model = diagnosis_model.Simple_Net(all_abnormity_num, label_levels).to(device)
    trained_model = torch.load(os.path.join(folder_path, wt_name + ".pth"))
    model.load_state_dict(trained_model["state_dict"])
    
    start_time = time.time()
    
    for label in range(label_levels):
        for i in range(class_size):
            output = utils.get_mr(device, disease, label, o_mr, f_mr)
            test_data[label * class_size + i] = output
            test_label[label * class_size + i] = label
    test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True)
    
    conf_mat = np.zeros((label_levels, label_levels), dtype=int)
    model.eval()
    corrects = 0
    total = len(test_loader.dataset)
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.to(device))
            _, predicted = torch.max(outputs, 1)
            labels = labels.to(device)
            corrects += (predicted == labels).sum().item()
            for i in range(len(labels)):
                conf_mat[labels[i]][predicted[i]] += 1
    acc = corrects / total
    time_elapsed = time.time() - start_time
    print(f"Time elapsed {time_elapsed // 60 :.0f}m {time_elapsed % 60 :.2f}s")
    print(f"Acc: {corrects}/{total} = {acc :.4f}")
    
    return conf_mat, corrects, total, acc

# test D1 and plot confusion matrices
def test(device):
    setting = utils.get_setting()
    diseases = setting.get_diseases(include_normal = False)
    folder_path = setting.D2_folder
    fig_folder = setting.fig_folder
    fig_file = "diagnosis1_confusion_matrix.pdf"
    fig_path = os.path.join(fig_folder, fig_file)
    
    fig = plt.figure(figsize = (12, 8))
    axs = fig.subplots(3, 4)
    
    row = 0
    col = 0
    
    with open(os.path.join(folder_path, "00D1TORS.csv"), "w", newline="") as file:  
        writer = csv.writer(file)  
        for disease in diseases:
            conf_mat, corrects, total, acc = get_conf_mat(device, disease)
            writer.writerow([disease, corrects, total, acc])
            mat = axs[row, col].matshow(conf_mat, cmap='Blues')
            max = np.max(conf_mat)
            for i in range(conf_mat.shape[0]):
                for j in range(conf_mat.shape[1]):
                    axs[row, col].text(j, i, conf_mat[i, j], ha='center', va='center', color='black' if conf_mat[i, j] < max / 2 else 'white')
            plt.colorbar(mat)
            if col == 3:
                col = 0
                row += 1
            else:
                col += 1
        
    plt.savefig(fig_path)
    plt.show()