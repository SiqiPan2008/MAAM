import os
import torch
import time
import random
import numpy as np
from Utils import utils
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from Diagnosis_Model import diagnosis_model
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def plot_tSNE(features, labels, diseases):
    setting = utils.get_setting()
    fig_folder = setting.fig_folder
    fig_file = "diagnosis2_tSNE.pdf"
    fig_path = os.path.join(fig_folder, fig_file)
    
    fig = plt.figure(figsize = (9, 8))
    
    tsne = TSNE(n_components=2)
    features_tsne = tsne.fit_transform(np.array(features))
    
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap='jet', alpha=0.5)
    plt.colorbar(scatter)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    
    plt.savefig(fig_path)
    plt.show()

def plot_all_ROC(labels, probs, diseases, draw_roc = True):
    setting = utils.get_setting()
    fig_folder = setting.fig_folder
    fig_file = "diagnosis2_ROC.pdf"
    fig_path = os.path.join(fig_folder, fig_file)
    
    if draw_roc:
        fig = plt.figure(figsize = (8, 8))
        axs = fig.subplots(4, 4)
    
    row = 0
    col = 0
    aucs = []
    
    for disease in diseases:
        fpr, tpr, _ = roc_curve(labels[disease], probs[disease])
        aucs.append(auc(fpr, tpr))
        if draw_roc:
            axs[row, col].plot(fpr, tpr, color = "red")
            axs[row, col].set_xlim([0.0, 1.0])
            axs[row, col].set_ylim([0.0, 1.0])
            axs[row, col].plot([0, 1], [0, 1], color = "black")
        if col == 3:
            col = 0
            row += 1
        else:
            col += 1
    
    for col in range(1, 4):
        axs[3, col].set_axis_off()
    
    if draw_roc:
        plt.savefig(fig_path)
        plt.show()
    return aucs

def plot_conf_mat(conf_mat, diseases):
    setting = utils.get_setting()
    fig_folder = setting.fig_folder
    fig_file = "diagnosis2_confusion_matrix.pdf"
    fig_path = os.path.join(fig_folder, fig_file)
    
    mat = plt.matshow(conf_mat, cmap='Blues')
    max = np.max(conf_mat)
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            plt.text(j, i, conf_mat[i, j], ha='center', va='center', color='black' if conf_mat[i, j] < max / 2 else 'white')
    plt.colorbar(mat)
    
    plt.savefig(fig_path)
    plt.show()

def plot_disease_tSNE_and_ROC_and_conf_mat(device, plt_tSNE = True, plt_ROC = True, plt_conf_mat = True):
    setting = utils.get_setting()
    name = "000D2"
    d1_folder = setting.D1_folder
    batch_size = setting.test_batch_size
    class_size = setting.D2_test_class_size
    abnormity_folder_path = setting.A_folder
    o_class_size = setting.O_test_class_size
    f_class_size = setting.F_test_class_size
    tSNE_point_num = setting.tSNE_point_num
    wt_name = setting.get_d_wt_file_name(name)
    o_mr_name = setting.get_o_tomr_name(setting.best_o_net)
    f_mr_name = setting.get_f_tomr_name(setting.best_f_net)
    folder_path = setting.get_folder_path(name)
    d2_input_length = setting.get_d2_input_length()
    diseases = setting.get_diseases(include_normal = False)
    o_abnormity_num = setting.get_abnormity_num("OCT Abnormities")
    f_abnormity_num = setting.get_abnormity_num("Fundus Abnormities")
    all_abnormity_num = setting.get_abnormity_num("All Abnormities")
    diseases_including_normal = setting.get_diseases(include_normal = True)
    disease_num_including_normal = setting.get_disease_num(include_normal = True)
    
    test_data = torch.zeros(disease_num_including_normal * class_size, d2_input_length)
    test_label = torch.zeros(disease_num_including_normal * class_size, dtype=torch.long)
    
    o_mr_path = os.path.join(abnormity_folder_path, o_mr_name + ".bin")
    o_mr = np.fromfile(o_mr_path, dtype = np.float64).reshape((o_abnormity_num, o_class_size, o_abnormity_num))
    f_mr_path = os.path.join(abnormity_folder_path, f_mr_name + ".bin")
    f_mr = np.fromfile(f_mr_path, dtype = np.float64).reshape((f_abnormity_num, f_class_size, f_abnormity_num))
    
    model = diagnosis_model.Simple_Net(d2_input_length, disease_num_including_normal).to(device)
    trained_model = torch.load(os.path.join(folder_path, wt_name + ".pth"))
    model.load_state_dict(trained_model["state_dict"])
    
    abnormity_num_models = {
        disease: diagnosis_model.Simple_Net(
            all_abnormity_num, 
            setting.get_disease_abnormity_num(disease, "All Abnormities") + 1
        ).to(device)
        for disease in diseases
    }
    for disease in abnormity_num_models.keys():
         trained_model = torch.load(os.path.join(d1_folder, setting.get_d1_single_disease_wt(disease) + ".pth"))
         abnormity_num_models[disease].load_state_dict(trained_model["state_dict"])
    
    start_time = time.time()
    for i in range(len(diseases_including_normal)):
        disease = diseases_including_normal[i]
        for j in range(class_size):
            output = utils.get_abnormity_nums_vector(device, disease, o_mr, f_mr, abnormity_num_models)
            output = output.detach()
            test_data[i * class_size + j] = output
            test_label[i * class_size + j] = i
    test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

    roc_labels = {disease: [] for disease in diseases_including_normal}
    roc_probs = {disease: [] for disease in diseases_including_normal}
    tSNE_features = [[] for _ in diseases_including_normal]
    tSNE_labels = [[] for _ in diseases_including_normal]
    conf_mat = np.zeros((disease_num_including_normal, disease_num_including_normal), dtype=int)
    
    model.eval()
    corrects = 0
    total = len(test_loader.dataset)
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.to(device))[0].cpu()
            _, predicted = torch.max(outputs, 0)
            corrects += predicted.cpu().item() == labels[0]
            
            conf_mat[labels[0]][predicted.item()] += 1
            tSNE_features[labels[0]].append(outputs)
            for i in range(len(diseases_including_normal)):
                disease = diseases_including_normal[i]
                roc_labels[disease].append(i == labels[0])
                roc_probs[disease].append(outputs[i])
            
    acc = corrects / total
    time_elapsed = time.time() - start_time
    print(f"Time elapsed {time_elapsed // 60 :.0f}m {time_elapsed % 60 :.2f}s")
    print(f"acc: {corrects}/{total} = {acc :.4f}")
    
    for i in range(len(tSNE_features)):
        tSNE_features[i] = random.sample(tSNE_features[i], tSNE_point_num)
        tSNE_labels[i] = [i for _ in range(tSNE_point_num)]
    tSNE_features = np.concatenate(tSNE_features)
    tSNE_labels = np.concatenate(tSNE_labels)
    
    if plt_tSNE:
        plot_tSNE(tSNE_features, tSNE_labels, diseases_including_normal)
    if plt_ROC:
        plot_all_ROC(roc_labels, roc_probs, diseases_including_normal)
    if plt_conf_mat:
        plot_conf_mat(conf_mat, diseases_including_normal)