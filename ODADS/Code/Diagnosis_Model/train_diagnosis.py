import os
import torch
from torch import nn
import torch.optim as optim
import time
import random
import copy
from PIL import Image
import csv
import numpy as np
from Abnormity_Models import abnormity_models
from Utils import utils
from Diagnosis_Model import diagnosis_model
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def train_abnormity_num_model(device, names):
    setting = utils.get_setting()
    name = names[0]
    LR = setting.LR
    batch_size = setting.batch_size
    class_size = setting.D1_train_class_size
    abnormity_folder_path = setting.A_folder
    o_class_size = setting.O_train_class_size
    f_class_size = setting.F_train_class_size
    wt_name = setting.get_d_wt_file_name(name)
    o_mr_name = setting.get_o_mr_name(names[1])
    f_mr_name = setting.get_f_mr_name(names[0])
    num_epochs = setting.get_num_epochs(name)
    folder_path = setting.get_folder_path(name)
    o_abnormity_num = setting.get_abnormity_num("OCT Abnormities")
    f_abnormity_num = setting.get_abnormity_num("Fundus Abnormities")
    all_abnormity_num = setting.get_abnormity_num("All Abnormities")
    
    label_levels = setting.get_disease_abnormity_num(name, "All Abnormities") + 1
    
    o_mr_path = os.path.join(abnormity_folder_path, o_mr_name + ".bin")
    o_mr = np.fromfile(o_mr_path, dtype = np.float64).reshape((o_abnormity_num, o_class_size, o_abnormity_num))
    f_mr_path = os.path.join(abnormity_folder_path, f_mr_name + ".bin")
    f_mr = np.fromfile(f_mr_path, dtype = np.float64).reshape((f_abnormity_num, f_class_size, f_abnormity_num))
    
    train_size = int(0.8 * class_size)
    valid_size = class_size - train_size
    train_data = torch.zeros(label_levels * train_size, all_abnormity_num)
    train_label = torch.zeros(label_levels * train_size, dtype=torch.long)
    valid_data = torch.zeros(label_levels * valid_size, all_abnormity_num)
    valid_label = torch.zeros(label_levels * valid_size, dtype=torch.long)
    
    model = diagnosis_model.Simple_Net(all_abnormity_num, label_levels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 7, gamma = 0.1)
    
    start_time = time.time()
    best_acc = 0.0
    LRs = [optimizer.param_groups[0]["lr"]]
    
    train_acc_history = []
    valid_acc_history = []
    train_losses = []
    valid_losses = []
    for epoch in range(num_epochs):
        for label in range(label_levels):
            for i in range(train_size):
                output = utils.get_mr(device, name, label, o_mr, f_mr)
                train_data[label * train_size + i] = output
                train_label[label * train_size + i] = label
        for label in range(label_levels):
            for i in range(valid_size):
                output = utils.get_mr(device, name, label, o_mr, f_mr)
                valid_data[label * valid_size + i] = output
                valid_label[label * valid_size + i] = label
        train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
        valid_dataset = torch.utils.data.TensorDataset(valid_data, valid_label)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch_size, shuffle = True)
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        model.train()
        running_loss = 0.0
        corrects = 0
        total = len(train_loader.dataset)
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            corrects += (predicted == labels.to(device)).sum().item()
        acc = corrects / total
        train_acc_history.append(acc)
        loss = running_loss / total
        train_losses.append(loss)
        time_elapsed = time.time() - start_time
        print(f"Time elapsed {time_elapsed // 60 :.0f}m {time_elapsed % 60 :.2f}s")
        print(f"Train loss: {loss :.4f}, acc: {acc :.4f}")
        
        model.eval()
        corrects = 0
        total = len(valid_loader.dataset)
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                outputs = model(inputs.to(device))
                loss = criterion(outputs, labels.to(device))
                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                corrects += (predicted == labels.to(device)).sum().item()
        acc = corrects / total
        valid_acc_history.append(acc)
        loss = val_running_loss / total
        valid_losses.append(loss)
        time_elapsed = time.time() - start_time
        print(f"Time elapsed {time_elapsed // 60 :.0f}m {time_elapsed % 60 :.2f}s")
        print(f"Valid loss: {loss :.4f}, acc: {acc :.4f}")
        
        if acc >= best_acc:
            best_acc = acc
            state = {
                "state_dict": model.state_dict(),
                "best_acc": best_acc,
                "optimizer": optimizer.state_dict()
            }
            torch.save(state, os.path.join(folder_path, wt_name + ".pth"))
            print(f"Data successfully written into {wt_name}.pth")
        
        scheduler.step()
        LRs.append(optimizer.param_groups[0]["lr"])
        print(f"optimizer learning rate: {LRs[-1]:.7f}")
        print()
    
    time_elapsed = time.time() - start_time
    print(f"training complete in {time_elapsed // 60 :.0f}m {time_elapsed % 60 :.2f}s")
    print(f"best valid acc: {best_acc :.4f}")
    
    return train_acc_history, valid_acc_history, train_losses, valid_losses, LRs





        
def train_disease_prob_model(device, names):
    setting = utils.get_setting()
    name = names[0]
    LR = setting.LR
    d1_folder = setting.D1_folder
    batch_size = setting.batch_size
    class_size = setting.D2_train_class_size
    abnormity_folder_path = setting.A_folder
    o_class_size = setting.O_train_class_size
    f_class_size = setting.F_train_class_size
    wt_name = setting.get_d_wt_file_name(name)
    o_mr_name = setting.get_o_mr_name(names[1])
    f_mr_name = setting.get_f_mr_name(names[0])
    num_epochs = setting.get_num_epochs(name)
    folder_path = setting.get_folder_path(name)
    d2_input_length = setting.get_d2_input_length()
    diseases = setting.get_diseases(include_normal = False)
    o_abnormity_num = setting.get_abnormity_num("OCT Abnormities")
    f_abnormity_num = setting.get_abnormity_num("Fundus Abnormities")
    all_abnormity_num = setting.get_abnormity_num("All Abnormities")
    diseases_including_normal = setting.get_diseases(include_normal = True)
    disease_num_including_normal= setting.get_disease_num(include_normal = True)

    train_size = int(0.8 * class_size)
    valid_size = class_size - train_size
    train_data = torch.zeros(disease_num_including_normal * train_size, d2_input_length)
    train_label = torch.zeros(disease_num_including_normal * train_size, dtype=torch.long)
    valid_data = torch.zeros(disease_num_including_normal * valid_size, d2_input_length)
    valid_label = torch.zeros(disease_num_including_normal * valid_size, dtype=torch.long)
    
    o_mr_path = os.path.join(abnormity_folder_path, o_mr_name + ".bin")
    o_mr = np.fromfile(o_mr_path, dtype = np.float64).reshape((o_abnormity_num, o_class_size, o_abnormity_num))
    f_mr_path = os.path.join(abnormity_folder_path, f_mr_name + ".bin")
    f_mr = np.fromfile(f_mr_path, dtype = np.float64).reshape((f_abnormity_num, f_class_size, f_abnormity_num))
    
    model = diagnosis_model.Simple_Net(d2_input_length, disease_num_including_normal).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 7, gamma = 0.1)
    
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
    best_acc = 0.0
    LRs = [optimizer.param_groups[0]["lr"]]
    
    train_acc_history = []
    valid_acc_history = []
    train_losses = []
    valid_losses = []
    for epoch in range(num_epochs):
        for i in range(disease_num_including_normal):
            disease = diseases_including_normal[i]
            for j in range(train_size):
                output = utils.get_abnormity_nums_vector(device, disease, o_mr, f_mr, abnormity_num_models)
                output = output.detach()
                train_data[i * train_size + j] = output
                train_label[i * train_size + j] = i
        for i in range(disease_num_including_normal):
            disease = diseases_including_normal[i]
            for j in range(valid_size):
                output = utils.get_abnormity_nums_vector(device, disease, o_mr, f_mr, abnormity_num_models)
                output = output.detach()
                valid_data[i * valid_size + j] = output
                valid_label[i * valid_size + j] = i
        train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
        valid_dataset = torch.utils.data.TensorDataset(valid_data, valid_label)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch_size, shuffle = True)
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        model.train()
        running_loss = 0.0
        corrects = 0
        total = len(train_loader.dataset)
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            corrects += (predicted == labels.to(device)).sum().item()
        acc = corrects / total
        train_acc_history.append(acc)
        running_loss = running_loss / total
        train_losses.append(running_loss)
        time_elapsed = time.time() - start_time
        print(f"Time elapsed {time_elapsed // 60 :.0f}m {time_elapsed % 60 :.2f}s")
        print(f"Train loss: {running_loss :.4f}, acc: {acc :.4f}")
        
        model.eval()
        corrects = 0
        total = len(valid_loader.dataset)
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                outputs = model(inputs.to(device))
                loss = criterion(outputs, labels.to(device))
                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                corrects += (predicted == labels.to(device)).sum().item()
        acc = corrects / total
        valid_acc_history.append(acc)
        val_running_loss = val_running_loss / total
        valid_losses.append(val_running_loss)
        time_elapsed = time.time() - start_time
        print(f"Time elapsed {time_elapsed // 60 :.0f}m {time_elapsed % 60 :.2f}s")
        print(f"Valid loss: {val_running_loss :.4f}, acc: {acc :.4f}")
        
        if acc >= best_acc:
            best_acc = acc
            state = {
                "state_dict": model.state_dict(),
                "best_acc": best_acc,
                "optimizer": optimizer.state_dict()
            }
            torch.save(state, os.path.join(folder_path, wt_name + ".pth"))
            print(f"Data successfully written into {wt_name}.pth")
        
        scheduler.step()
        LRs.append(optimizer.param_groups[0]["lr"])
        print(f"optimizer learning rate: {LRs[-1]:.7f}")
        print()
    
    time_elapsed = time.time() - start_time
    print(f"training complete in {time_elapsed // 60 :.0f}m {time_elapsed % 60 :.2f}s")
    print(f"best valid acc: {best_acc :.4f}")
    
    return train_acc_history, valid_acc_history, train_losses, valid_losses, LRs






def train_d(device, names):
    setting = utils.get_setting()
    name = names[0]
    rs_name = setting.get_d_rs_file_name(name)
    folder_path = setting.get_folder_path(name)
    
    if setting.is_diagnosis1(name):
        trainAccHistory, validAccHistory, trainLosses, validLosses, LRs = train_abnormity_num_model(device, names)
    else:
        trainAccHistory, validAccHistory, trainLosses, validLosses, LRs = train_disease_prob_model(device, names)
        
    with open(os.path.join(folder_path, rs_name + ".csv"), "w", newline="") as file:  
        writer = csv.writer(file)  
        for i in range(len(trainLosses)):  
            writer.writerow([i + 1, validAccHistory[i], trainAccHistory[i], validLosses[i], trainLosses[i], LRs[i]])
    print(f"Data successfully written into {rs_name}.csv")
    print("\n" for _ in range(3))
    
    utils.curve(validAccHistory, trainAccHistory, validLosses, trainLosses, os.path.join(folder_path, rs_name + ".pdf"))
    
def train(device, names):
    setting = utils.get_setting()
    if setting.is_diagnosis1(names[0]):
        disease_name = setting.get_disease_name(names[0])
        if disease_name:
            train_d(device, names)
        else:
            diseases = setting.get_diseases(include_normal = False)
            for disease in diseases:
                train_d(
                    device, 
                    [setting.get_d1_single_disease_rs(name, disease) for name in names]
                )
    else:
        train_d(device, names)