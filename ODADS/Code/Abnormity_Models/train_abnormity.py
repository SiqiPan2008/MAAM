import os
import torch
from torch import nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import Subset
import time
import csv
import numpy as np
from ODADS.Code.Utils import utils
from ODADS.Code.Abnormity_Models import abnormity_models
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def train(device, name):
    setting = utils.get_setting()
    LR = setting.LR
    batch_size = setting.batch_size
    cross_valid = setting.use_cross_valid
    feature_extract = setting.feature_extract
    save_model_frequency = setting.save_model_frequency
    net_name = setting.get_net(name)
    num_epochs = setting.get_num_epochs(name)
    img_folder = setting.get_img_folder(name)
    folder_path = setting.get_folder_path(name)
    temp_wts_folder = os.path.join(folder_path, wt_file_name)
    wt_file_name = setting.get_wt_file_name(name)
    rs_file_name = setting.get_rs_file_name(name)
    num_classes = setting.get_abnormity_num(name)
    use_pretrained = setting.is_transfer_learning(name)
    is_transfer_learning = setting.is_transfer_learning(name)
    
    model = abnormity_models.initialize_abnormity_model(net_name, num_classes, feature_extract, use_pretrained)
    model = model.to(device)
    if not is_transfer_learning:
        trained_model = torch.load(os.path.join(folder_path, setting.get_transfer_learning_wt(name) + ".pth"))
        model.load_state_dict(trained_model['state_dict']) 
    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if feature_extract:
                params_to_update.append(param)
            print("\t", name)
            
    optimizer = optim.Adam(params_to_update, lr = LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 7, gamma = 0.1)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    last_time = start_time
    best_acc = [0]
    model.to(device)
    valid_acc_history = []
    train_acc_history = []
    valid_losses = []
    train_losses = []
    LRs = [optimizer.param_groups[0]["lr"]]
    
    data_transforms = transforms.Compose([transforms.ToTensor()])
    if cross_valid:
        image_dataset = datasets.ImageFolder(img_folder, data_transforms)
    else:
        image_datasets = {x: datasets.ImageFolder(os.path.join(img_folder, x), data_transforms) for x in ["train", "valid"]}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = batch_size, shuffle = True) for x in ["train", "valid"]}
    
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)
        
        if cross_valid:
            labels = np.array(image_dataset.targets)
            train_indices = []
            valid_indices = []
            for class_id in range(num_classes):
                class_indices = np.where(labels == class_id)[0]
                np.random.shuffle(class_indices)
                split_point = int(0.8 * len(class_indices))
                train_indices.extend(class_indices[:split_point])
                valid_indices.extend(class_indices[split_point:])
            train_dataset = Subset(image_dataset, train_indices)
            valid_dataset = Subset(image_dataset, valid_indices)
            dataloaders = {
                "train": torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True),
                "valid": torch.utils.data.DataLoader(valid_dataset, batch_size = batch_size, shuffle=True)
            }
    
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                        
                    preds = torch.max(outputs, 1)[1]
                    if phase == "train":
                        loss.to(device)
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            dataset_len = len(dataloaders[phase].dataset)
            epoch_loss = running_loss / dataset_len
            epoch_acc = running_corrects / dataset_len
            total_time_elapsed = time.time() - start_time
            time_elapsed = time.time() - last_time
            last_time = time.time()
            print(f"total time elapsed {total_time_elapsed // 60 :.0f}m {total_time_elapsed % 60 :.2f}s")
            print(f"time elapsed {time_elapsed // 60 :.0f}m {total_time_elapsed % 60 :.2f}s")
            print(f"{phase} loss: {epoch_loss :.4f}, acc: {epoch_acc :.4f}")
            
            if phase == "valid" and epoch_acc >= best_acc[-1]:
                best_acc[-1] = epoch_acc
                state = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict()
                }
                epoch_range_beginning = int(epoch / save_model_frequency) * save_model_frequency
                torch.save(state, os.path.join(temp_wts_folder, wt_file_name + f"{epoch_range_beginning: 3d}.pth"))
                print(f"Data successfully written into {wt_file_name}.pth")
            if phase == "valid" and (epoch + 1) % save_model_frequency == 0:
                best_acc.append(0)
                
            if phase == "valid":
                valid_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
                scheduler.step(epoch_loss)
            elif phase == "train":
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)
        
        print(f"optimizer learning rate: {optimizer.param_groups[0]['lr'] :.7f}")
        LRs.append(optimizer.param_groups[0]["lr"])
        print()
        
    total_time_elapsed = time.time() - start_time
    print(f"training complete in {total_time_elapsed // 60 :.0f}m {total_time_elapsed % 60 :.2f}s")
    print(f"best valid acc: {best_acc}")

    with open(os.path.join(folder_path, rs_file_name + ".csv"), "w", newline="") as file:  
        writer = csv.writer(file)  
        for i in range(len(train_losses)):  
            writer.writerow([i + 1, valid_acc_history[i].item(), train_acc_history[i].item(), valid_losses[i], train_losses[i], LRs[i]])
    print(f"Data successfully written into {rs_file_name}.csv")
    
    utils.curve(valid_acc_history, train_acc_history, valid_losses, train_losses, os.path.join(folder_path, rs_file_name + ".pdf"))