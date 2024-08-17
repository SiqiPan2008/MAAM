import os
import torch
import csv
import numpy as np
from torchvision import transforms, datasets
from PIL import Image
from Abnormity_Models import abnormity_models, classify_abnormity
from Utils import utils
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def test(device, name, filename):
    setting = utils.get_setting()
    use_top_probs = setting.use_top_probs
    feature_extract = setting.feature_extract
    A_top_probs_max_num = setting.A_top_probs_max_num
    A_top_probs_min_prob = setting.A_top_probs_min_prob
    net_name = setting.get_net(name)
    img_folder = setting.get_img_folder(name)
    folder_path = setting.get_folder_path(name)
    num_classes = setting.get_abnormity_num(name)
    wt_file_name = os.path.join(setting.get_wt_file_name(name), filename)
    
    model = abnormity_models.initialize_abnormity_model(net_name, num_classes, feature_extract)
    model = model.to(device)
    trained_model = torch.load(os.path.join(folder_path, wt_file_name))
    model.load_state_dict(trained_model["state_dict"])
    abnormities = setting.get_abnormities(name)
    class_to_idx = {abnormities[i][1]: i for i in range(len(abnormities))}
    
    corrects = 0
    total = 0
    for abnormity in abnormities:
        abnormity = abnormity[1]
        abnormity_folder = os.path.join(img_folder, abnormity)
        for img_name in os.listdir(abnormity_folder):
            img = Image.open(os.path.join(abnormity_folder, img_name))
            output = classify_abnormity.get_abnormities_probs(img, model, device)
            if use_top_probs:
                top_indices = utils.get_top_prob_indices(output, A_top_probs_max_num, A_top_probs_min_prob)
                corrects += class_to_idx[abnormity] in top_indices
            else:
                _, pred = torch.max(output, dim=0)
                corrects += pred == class_to_idx[abnormity]
            total += 1
        print(f"completed {abnormity}")
    accuracy = corrects / total
    print(f"corrects: {corrects}/{total}")
    print(f"accuracy: {accuracy}")
    return corrects, total, accuracy

def test_multiple(device, name):
    setting = utils.get_setting()
    folder_path = setting.get_folder_path(name)
    temp_wts_folder = os.path.join(folder_path, setting.get_wt_file_name(name))
    
    test_acc = []
    epoch_nums = []
    with open(os.path.join(folder_path, name + ".csv"), "w", newline="") as file:  
        writer = csv.writer(file)
        for filename in os.listdir(temp_wts_folder):
            if os.path.splitext(filename)[1] == ".pth":
                corrects, total, accuracy = test(device, name, filename)
                epoch_num = setting.get_epoch_num(filename)
                test_acc.append(accuracy)
                epoch_nums.append(epoch_num)
            writer.writerow([epoch_num, corrects, total, accuracy])

        valid_acc = []
        with open(os.path.join(folder_path, setting.get_training_rs_name(name) + ".csv"), mode='r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if int(row[0]) in epoch_nums:
                    valid_acc.append(float(row[3]))
        
        test_acc = np.array(test_acc)
        valid_acc = np.array(valid_acc)
        train_size, test_size = setting.get_abnormity_model_datasizes(name)
        valid_size = train_size - int(train_size * 0.8)
        combined_acc = test_acc * (test_size / (test_size + valid_size)) + \
            valid_acc * (valid_size / (test_size + valid_size))
        best_combined_acc = np.max(combined_acc)
        best_index = np.argmax(combined_acc)
        best_epoch = epoch_nums[best_index]
        writer.writerow([best_epoch])
        
    print(f"best epoch: {best_epoch}, combined acc = {best_combined_acc}")
    utils.rename_file_by_path(
        os.path.join(temp_wts_folder, setting.get_wt_file_name(name) + f"{best_epoch: 3d}.pth"),
        os.path.join(folder_path, setting.get_wt_file_name(name) + ".pth")
    )
    for filename in os.listdir(temp_wts_folder):
        os.remove(os.path.join(temp_wts_folder, filename))
    os.rmdir(temp_wts_folder)
    print(f"Data successfully written into {name}.csv")
    
    return best_combined_acc

def get_final_abnormity_model(name, t_acc, f_acc):
    setting = utils.get_setting()
    wt_name = setting.get_wt_file_name(name)
    folder_path = setting.get_folder_path(name)
    delete_ver = "F" if t_acc > f_acc else "T"
    rename_ver = "T" if t_acc > f_acc else "F"
    os.remove(os.path.join(folder_path, wt_name + f" - {delete_ver}.pth"))
    os.rename(
        os.path.join(folder_path, wt_name + f" - {rename_ver}.pth"),
        os.path.join(folder_path, wt_name + ".pth")
    )
    
def get_model_results(device, name):
    setting = utils.get_setting()
    batch_size = setting.test_batch_size
    feature_extract = setting.feature_extract
    net_name = setting.get_net(name)
    img_folder = setting.get_img_folder(name)
    folder_path = setting.get_folder_path(name)
    num_classes = setting.get_abnormity_num(name)
    wt_file_name = setting.get_wt_file_name(name)
    
    model = abnormity_models.initialize_abnormity_model(net_name, num_classes, feature_extract)
    model = model.to(device)
    trained_model = torch.load(os.path.join(folder_path, wt_file_name + ".pth"))
    model.load_state_dict(trained_model["state_dict"])
    model.eval()
    transform = transforms.Compose([
                    transforms.Lambda(lambda x: utils.resize_long_edge(x)),
                    transforms.ToTensor()
                ])
    image_dataset = datasets.ImageFolder(img_folder, transform)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size = batch_size, shuffle=True)
    
    outputs = [[] for _ in range(num_classes)]
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            batch_outputs = model(inputs)
            output = torch.nn.functional.softmax(batch_outputs[0], dim = 0)
            outputs[labels[0]].append(output.cpu().detach().tolist())
            
    np.array(outputs).tofile(os.path.join(folder_path, name + ".bin"))
    print(f"model results saved to {name}.bin")