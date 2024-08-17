import os
import torch
import csv
import numpy as np
from torchvision import transforms, datasets
from PIL import Image
from ODADS.Code.Abnormity_Models import abnormity_models, classify_abnormity
from ODADS.Code.Utils import utils
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def test_acc(device, name, filename):
    setting = utils.get_setting()
    use_top_probs = setting.use_top_probs
    feature_extract = setting.feature_extract
    net_name = setting.get_net(name)
    img_folder = setting.get_img_folder(name)
    folder_path = setting.get_folder_path(name)
    num_classes = setting.get_abnormities_num(name)
    wt_file_name = os.path.join(setting.get_wt_file_name(name), filename)
    
    model = abnormity_models.initialize_abnormity_model(net_name, num_classes, feature_extract)
    model = model.to(device)
    trained_model = torch.load(os.path.join(folder_path, wt_file_name + ".pth"))
    model.load_state_dict(trained_model["state_dict"])
    abnormities = setting.get_abnormities(name)
    class_to_idx = {abnormities[i]: i for i in range(len(abnormities))}
    
    corrects = 0
    total = 0
    for abnormity in abnormities:
        abnormity_folder = os.path.join(img_folder, abnormity)
        for img_name in os.listdir(abnormity_folder):
            img = Image.open(os.path.join(abnormity_folder, img_name))
            output = classify_abnormity.get_abnormities_probs(img, model, device)
            if use_top_probs:
                top_indices = utils.get_top_prob_indices(output, 4, 0.9)
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

def test_multiple_acc(device, name):
    setting = utils.get_setting()
    folder_path = setting.get_folder_path(name)
    temp_wts_folder = os.path.join(folder_path, setting.get_wt_file_name(name))
    
    with open(os.path.join(folder_path, name + ".csv"), "w", newline="") as file:  
        writer = csv.writer(file)
        for filename in os.listdir(temp_wts_folder):
            if os.path.splitext(filename)[1] == ".pth":
                corrects, total, accuracy = test_acc(device, name, filename)
            writer.writerow([filename, corrects, total, accuracy])
    print(f"Data successfully written into {filename}.csv")
    
    
    
def get_model_results(device, name):
    setting = utils.get_setting()
    feature_extract = setting.feature_extract
    net_name = setting.get_net(name)
    img_folder = setting.get_img_folder(name)
    folder_path = setting.get_folder_path(name)
    num_classes = setting.get_abnormities_num(name)
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
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size = 1, shuffle=True)
    
    outputs = [[] for _ in range(num_classes)]
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            batch_outputs = model(inputs)
            output = torch.nn.functional.softmax(batch_outputs[0], dim = 0)
            outputs[labels[0]].append(output.cpu().detach().tolist())
            
    np.array(outputs).tofile(os.path.join(folder_path, name + ".bin"))