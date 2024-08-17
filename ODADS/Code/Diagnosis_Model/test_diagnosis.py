import os
import torch
import time
import numpy as np
from Utils import utils
from Diagnosis_Model import diagnosis_model, train_diagnosis
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
def test_abnormity_num_model(device, name):
    setting = utils.get_setting()
    batch_size = setting.test_batch_size
    class_size = setting.D1_train_class_size
    abnormity_folder_path = setting.A_folder
    o_class_size = setting.O_train_class_size
    f_class_size = setting.F_train_class_size
    wt_name = setting.get_wt_name(name)
    o_mr_name = setting.get_o_mr_name(name)
    f_mr_name = setting.get_f_mr_name(name)
    folder_path = setting.get_folder_path(name)
    o_abnormity_num = setting.get_abnormity_num("OCT Abnormities")
    f_abnormity_num = setting.get_abnormity_num("Fundus Abnormities")
    all_abnormity_num = setting.get_abnormity_num("All Abnormities")
    label_levels = setting.get_disease_abnormity_num(name, "All Abnormities") + 1
    
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
            output = utils.get_mr(device, name, label, o_mr, f_mr)
            test_data[label * class_size + i] = output
            test_label[label * class_size + i] = label
    test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True)
    
    model.eval()
    corrects = 0
    total = len(test_loader.dataset)
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.to(device))
            _, predicted = torch.max(outputs, 1)
            corrects += (predicted == labels.to(device)).sum().item()
    acc = corrects / total
    time_elapsed = time.time() - start_time
    print(f"Time elapsed {time_elapsed // 60 :.0f}m {time_elapsed % 60 :.2f}s")
    print(f"Acc: {corrects}/{total} = {acc :.4f}")
    
    return acc

        
def test_disease_prob_model(device, name):
    setting = utils.get_setting()
    d1_folder = setting.D1_folder
    batch_size = setting.batch_size
    class_size = setting.D2_train_class_size
    abnormity_folder_path = setting.A_folder
    o_class_size = setting.O_train_class_size
    f_class_size = setting.F_train_class_size
    wt_name = setting.get_wt_name(name)
    o_mr_name = setting.get_o_mr_name(name)
    f_mr_name = setting.get_f_mr_name(name)
    folder_path = setting.get_folder_path(name)
    d2_input_length = setting.get_d2_input_length()
    diseases = setting.get_diseases(include_normal = False)
    o_abnormity_num = setting.get_abnormity_num("OCT Abnormities")
    f_abnormity_num = setting.get_abnormity_num("Fundus Abnormities")
    all_abnormity_num = setting.get_abnormity_num("All Abnormities")
    diseases_including_normal = setting.get_diseases(include_normal = True)
    disease_num_including_normal= setting.get_disease_num(include_normal = True)
    
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
         trained_model = torch.load(os.path.join(d1_folder, setting.get_d1_single_disease_wt(name, disease) + ".pth"))
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

    model.eval()
    corrects = 0
    total = len(test_loader.dataset)
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.to(device))
            _, predicted = torch.max(outputs, 1)
            corrects += (predicted.cpu() == labels).sum().item()
    acc = corrects / total
    time_elapsed = time.time() - start_time
    print(f"Time elapsed {time_elapsed // 60 :.0f}m {time_elapsed % 60 :.2f}s")
    print(f"acc: {corrects}/{total} = {acc :.4f}")
    
    return acc


def test(device, name):
    setting = utils.get_setting()
    disease_name = setting.get_disease_name(name)
    if setting.is_diagnosis1(name):
        if disease_name:
            acc = test_abnormity_num_model(device, name)
        else:
            acc = []
            diseases = setting.get_diseases(include_normal = False)
            for disease in diseases:
                acc.append(test_abnormity_num_model(device, setting.get_d1_single_disease_rs(name, disease)))
    else:
        acc = test_disease_prob_model(device, name)
        
    print(acc)
