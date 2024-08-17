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
from ODADS.Code.Abnormity_Models import abnormity_models
from ODADS.Code.Utils import utils
from ODADS.Code.Diagnosis_Model import diagnosis_model
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def get_mr(device, name, label, o_mr, f_mr):
    setting = utils.get_setting()
    o_abnormities = setting.get_abnormities("OCT Abnormities")
    f_abnormities = setting.get_abnormities("Fundus Abnormities")
    o_abnormity_num = setting.get_abnormity_num("OCT Abnormities")
    f_abnormity_num = setting.get_abnormity_num("Fundus Abnormities")
    correct_abnormities = setting.get_correct_abnormities(name)
    incorrect_abnormities = setting.get_correct_abnormities(name)
     
    selCorrectAbnormities = random.sample(correct_abnormities, label)
    selIncorrectAbnormities = random.sample(incorrect_abnormities, o_abnormity_num + f_abnormity_num - label)
    selAbnormities = selCorrectAbnormities + selIncorrectAbnormities
    
    oOutput = torch.zeros([o_abnormity_num])
    fOutput = torch.zeros([f_abnormity_num])
    for abnormity in selAbnormities:
        abnormityType, abnormityName = abnormity[0], abnormity[1]
        if abnormityType == "OCT":
            output = random.choice(o_mr[o_abnormities.index(abnormityName)])
            oOutput = torch.max(torch.tensor(output), oOutput)
        elif abnormityType == "Fundus":
            output = random.choice(f_mr[f_abnormities.index(abnormityName)])
            fOutput = torch.max(torch.tensor(output), fOutput)
    output = torch.concat([fOutput, oOutput]).to(device)
    return output

def train_abnormity_num_model(device, name):
    setting = utils.get_setting()
    LR = setting.LR
    batch_size = setting.batch_size
    class_size = setting.D1_train_class_size
    abnormity_folder_path = setting.A_folder
    o_class_size = setting.O_train_class_size
    f_class_size = setting.F_train_class_size
    o_mr_name = setting.get_o_mr_name(name)
    f_mr_name = setting.get_f_mr_name(name)
    num_epochs = setting.get_num_epochs(name)
    folder_path = setting.get_folder_path(name)
    disease_name = setting.get_disease_name(name)
    allAbnormities = setting.get_abnormities("All Abnormities")
    o_abnormity_num = setting.get_abnormity_num("OCT Abnormities")
    f_abnormity_num = setting.get_abnormity_num("Fundus Abnormities")
    all_abnormity_num = setting.get_abnormity_num("All Abnormities")
    
    label_levels = o_abnormity_num + f_abnormity_num + 1
    
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
    
    model = diagnosis_model.Simple_Net(all_abnormity_num, label_levels)
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
        for grade in range(label_levels):
            for i in range(train_size):
                output = get_mr(allAbnormities, disease_name, o_abnormity_num, f_abnormity_num, grade, o_mr, f_mr)
                train_data[grade * train_size + i] = output
                train_label[grade * train_size + i] = grade
        for grade in range(label_levels):
            for i in range(valid_size):
                output = get_mr(allAbnormities, disease_name, o_abnormity_num, f_abnormity_num, grade, o_mr, f_mr)
                valid_data[grade * valid_size + i] = output
                valid_label[grade * valid_size + i] = grade
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
            torch.save(state, os.path.join(folder_path, name + ".pth"))
            print(f"Data successfully written into {name}.pth")
        
        scheduler.step()
        LRs.append(optimizer.param_groups[0]["lr"])
        print(f"optimizer learning rate: {LRs[-1]:.7f}")
        print()
    
    time_elapsed = time.time() - start_time
    print(f"training complete in {time_elapsed // 60 :.0f}m {time_elapsed % 60 :.2f}s")
    print(f"best valid acc: {best_acc :.4f}")
    
    return train_acc_history, valid_acc_history, train_losses, valid_losses, LRs






def get_abnormity_nums_vector(device, name, o_mr, f_mr, abnormity_num_models):
    setting = utils.get_setting()
    disease_abnormity_num = setting.get_disease_abnormity_num("All Abnormities")
    
    abnormity_output = get_mr(device, name, disease_abnormity_num, o_mr, f_mr)
    abnormity_output = abnormity_output.unsqueeze(0).type(torch.float32).to(device)
    abnormity_nums_vector = torch.empty([0]).to(device)
    for disease in abnormity_num_models.keys():
        temp_model = abnormity_num_models[disease]
        temp_model.eval()
        abnormity_num_output = temp_model(abnormity_output)[0]
        abnormity_num_output = torch.nn.functional.softmax(abnormity_num_output, 0)
        abnormity_nums_vector = torch.cat([abnormity_nums_vector, abnormity_num_output])
    return abnormity_nums_vector
        
def train_disease_prob_model(device, name):
    setting = utils.get_setting()
    LR = setting.LR
    d1_folder = setting.D1_folder
    batch_size = setting.batch_size
    class_size = setting.D2_train_class_size
    abnormity_folder_path = setting.A_folder
    o_class_size = setting.O_train_class_size
    f_class_size = setting.F_train_class_size
    o_mr_name = setting.get_o_mr_name(name)
    f_mr_name = setting.get_f_mr_name(name)
    num_epochs = setting.get_num_epochs(name)
    folder_path = setting.get_folder_path(name)
    d2_input_length = setting.get_d2_input_length()
    diseases = setting.get_diseases(include_normal = False)
    allAbnormities = setting.get_abnormities("All Abnormities")
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
            setting.get_disease_abnormity_num("All Abnormities")
        ).to(device)
        for disease in diseases
    }
    for disease in abnormity_num_models.keys():
         trained_model = torch.load(os.path.join(d1_folder, setting.get_d1_single_disease_wt(name, disease)))
         abnormity_num_models[disease].load_state_dict(trained_model["state_dict"])
    
    startTime = time.time()
    bestAcc = 0.0
    LRs = [optimizer.param_groups[0]["lr"]]
    
    trainAccHistory = []
    validAccHistory = []
    trainLosses = []
    validLosses = []
    for epoch in range(num_epochs):
        for i in range(disease_num_including_normal):
            disease = diseases_including_normal[i]
            for j in range(train_size):
                output = get_abnormity_nums_vector(device, disease, o_mr, f_mr, allAbnormities, abnormity_num_models)
                output = output.detach()
                train_data[i * train_size + j] = output
                train_label[i * train_size + j] = i
        for i in range(disease_num_including_normal):
            disease = diseases_including_normal[i]
            for j in range(valid_size):
                output = get_abnormity_nums_vector(device, disease, o_mr, f_mr, allAbnormities, abnormity_num_models)
                output = output.detach()
                valid_data[i * valid_size + j] = output
                valid_label[i * valid_size + j] = i
        trainDataset = torch.utils.data.TensorDataset(train_data, train_label)
        validDataset = torch.utils.data.TensorDataset(valid_data, valid_label)
        trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size = batch_size, shuffle = True)
        validLoader = torch.utils.data.DataLoader(validDataset, batch_size = batch_size, shuffle = True)
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        model.train()
        runningLoss = 0.0
        corrects = 0
        total = len(trainLoader.dataset)
        for inputs, labels in trainLoader:
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            runningLoss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            corrects += (predicted == labels.to(device)).sum().item()
        acc = corrects / total
        trainAccHistory.append(acc)
        runningLoss = runningLoss / total
        trainLosses.append(runningLoss)
        timeElapsed = time.time() - startTime
        print(f"Time elapsed {timeElapsed // 60 :.0f}m {timeElapsed % 60 :.2f}s")
        print(f"Train loss: {runningLoss :.4f}, acc: {acc :.4f}")
        
        model.eval()
        corrects = 0
        total = len(validLoader.dataset)
        valRunningLoss = 0.0
        with torch.no_grad():
            for inputs, labels in validLoader:
                outputs = model(inputs.to(device))
                loss = criterion(outputs, labels.to(device))
                valRunningLoss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                corrects += (predicted == labels.to(device)).sum().item()
        acc = corrects / total
        validAccHistory.append(acc)
        valRunningLoss = valRunningLoss / total
        validLosses.append(valRunningLoss)
        timeElapsed = time.time() - startTime
        print(f"Time elapsed {timeElapsed // 60 :.0f}m {timeElapsed % 60 :.2f}s")
        print(f"Valid loss: {valRunningLoss :.4f}, acc: {acc :.4f}")
        
        if acc >= bestAcc:
            bestAcc = acc
            state = {
                "state_dict": model.state_dict(),
                "best_acc": bestAcc,
                "optimizer": optimizer.state_dict()
            }
            torch.save(state, os.path.join(folder_path, name + ".pth"))
            print(f"Data successfully written into {name}.pth")
        
        scheduler.step()
        LRs.append(optimizer.param_groups[0]["lr"])
        print(f"optimizer learning rate: {LRs[-1]:.7f}")
        print()
    
    timeElapsed = time.time() - startTime
    print(f"training complete in {timeElapsed // 60 :.0f}m {timeElapsed % 60 :.2f}s")
    print(f"best valid acc: {bestAcc :.4f}")
    
    return trainAccHistory, validAccHistory, trainLosses, validLosses, LRs






def train(device, name):
    setting = utils.get_setting()
    disease_name = setting.get_disease_name(name)
    folder_path = setting.get_folder_path(name)
    
    if disease_name:
        trainAccHistory, validAccHistory, trainLosses, validLosses, LRs = train_abnormity_num_model(device, name)
    else:
        trainAccHistory, validAccHistory, trainLosses, validLosses, LRs = train_disease_prob_model(device, name)
        
    with open(os.path.join(folder_path, name + ".csv"), "w", newline="") as file:  
        writer = csv.writer(file)  
        for i in range(len(trainLosses)):  
            writer.writerow([i + 1, validAccHistory[i], trainAccHistory[i], validLosses[i], trainLosses[i], LRs[i]])
    print(f"Data successfully written into {name}.csv")
    print("\n" for _ in range(3))
    
    utils.curve(validAccHistory, trainAccHistory, validLosses, trainLosses, os.path.join(folder_path, name + ".pdf"))