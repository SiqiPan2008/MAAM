import os
import torch
import csv
import numpy as np
from torchvision import transforms, datasets
from PIL import Image
from Utils import utils
from AbnormityModels import abnormityModel, classify
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def testAcc(device, name, filename):
    setting = utils.getSetting()
    use_top_probs = setting.use_top_probs
    feature_extract = setting.feature_extract
    net_name = setting.get_net(name)
    img_folder = setting.get_img_folder(name)
    folder_path = setting.get_folder_path(name)
    num_classes = setting.get_num_abnormities(name)
    wt_file_name = os.path.join(setting.get_wt_file_name(name), filename)
    
    model, _ = abnormityModel.initializeAbnormityModel(net_name, num_classes, feature_extract)
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
            output = classify.classify(img, model, device)
            if use_top_probs:
                top_indices = utils.getTopProbIndices(output, 4, 0.9)
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

def testMultipleAcc(device, name):
    setting = utils.getSetting()
    folder_path = setting.get_folder_path(name)
    temp_wts_folder = os.path.join(folder_path, setting.get_wt_file_name(name))
    
    with open(os.path.join(folder_path, name + ".csv"), "w", newline="") as file:  
        writer = csv.writer(file)
        for filename in os.listdir(temp_wts_folder):
            if os.path.splitext(filename)[1] == ".pth":
                corrects, total, accuracy = testAcc(device, name, filename)
            writer.writerow([filename, corrects, total, accuracy])
    print(f"Data successfully written into {filename}.csv")