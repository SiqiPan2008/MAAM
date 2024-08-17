from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import json
import random
import setting
from typing import Dict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# read Criteria.json and get criteria
def get_criteria():
    with open(r'ODADS\Code\Criteria.json', 'r', encoding='utf-8') as file:
        criteria = json.load(file)
    return criteria

# load json file
def load_json(filename: str) -> Dict:
    with open(filename, 'r') as file:
        return json.load(file)

# read Setting.json
def get_setting():
    json_data = load_json('ODADS/CODE/Setting.json')
    return setting.Setting.from_dict(json_data)

# resize the image so that its longer edge has length "longEdgeSize"
# then paste it on a square black background
def resize_long_edge(img, long_edge_size = 224):
    width, height = img.size
    if width > height:
        new_size = (long_edge_size, int(height * long_edge_size / width))
        loc = (0, int((long_edge_size - new_size[1]) / 2))
    else:
        new_size = (int(long_edge_size * width / height), long_edge_size)
        loc = (int((long_edge_size - new_size[0]) / 2), 0)
    img = img.resize(new_size)
    black_background = Image.new("RGB", (long_edge_size, long_edge_size), "black")
    black_background.paste(img, loc)
    return black_background

# use valid and train accuracies and losses to draw curves
def curve(valid_acc_history, train_acc_history, valid_losses, train_losses, rs_file_name, show = False):
    plt.clf()
    x = range(1, len(train_losses) + 1)
    
    vAH = [valid_acc_history[i] for i in range(len(x))]
    tAH = [train_acc_history[i] for i in range(len(x))]
    plt.subplot(2, 1, 1)
    plt.plot(x, vAH, label = "validAcc")
    plt.plot(x, tAH, label = "trainAcc")
    plt.title("Accuracy Curve")  
    plt.xlabel("Epoch")  
    plt.ylabel("Accuracy")  
    plt.legend()  
      
    plt.subplot(2, 1, 2)
    plt.plot(x, valid_losses, label = "validLoss")
    plt.plot(x, train_losses, label = "trainLoss")
    plt.title("Loss Curve")  
    plt.xlabel("Epoch")  
    plt.ylabel("Loss")  
    plt.legend() 
    
    plt.tight_layout()
    plt.savefig(rs_file_name, format="pdf")
    if show:
        plt.show()

# resizeLongEdge() an image and then convert it to Tensor
def resize_and_to_tensor(img, custom_resize = 224):
    if custom_resize != 0:
        img = resize_long_edge(img, long_edge_size = custom_resize)
    to_tensor = transforms.ToTensor()
    img = to_tensor(img)
    return img

# show an image in Matplotlib
def img_show(img, ax = None, title = None):
    if ax is None:
        fig, ax = plt.subplots()
    img = np.array(img).transpose((1, 2, 0))
    ax.imshow(img)
    ax.set_title(title)
    plt.show()

# get the least number of probabilities so that their sum is greater tham minScore
# returns the indices of the top probabilities
def get_top_prob_indices(output, allow_num, min_score):
    sorted_output, indices = torch.sort(output, descending=True)
    sum = 0
    top_indices = []
    for i in range(allow_num):
        if sum >= min_score:
            break
        sum += sorted_output[i]
        top_indices.append(indices[i])
    return top_indices

# read from model results and return the result of a random image of a certain disease
def get_mr(device, name, label, o_mr, f_mr):
    setting = get_setting()
    o_abnormities = setting.get_abnormities("OCT Abnormities")
    f_abnormities = setting.get_abnormities("Fundus Abnormities")
    o_abnormity_num = setting.get_abnormity_num("OCT Abnormities")
    f_abnormity_num = setting.get_abnormity_num("Fundus Abnormities")
    disease_total_abnormity_num = setting.get_disease_abnormity_num(name, "All Abnormities")
    correct_abnormities = setting.get_correct_abnormities(name)
    incorrect_abnormities = setting.get_incorrect_abnormities(name)
     
    selCorrectAbnormities = random.sample(correct_abnormities, label)
    selIncorrectAbnormities = random.sample(incorrect_abnormities, disease_total_abnormity_num - label)
    selAbnormities = selCorrectAbnormities + selIncorrectAbnormities
    
    oOutput = torch.zeros([o_abnormity_num])
    fOutput = torch.zeros([f_abnormity_num])
    for abnormity in selAbnormities:
        if abnormity[0] == "OCT":
            output = random.choice(o_mr[o_abnormities.index(abnormity)])
            oOutput = torch.max(torch.tensor(output), oOutput)
        elif abnormity[0] == "Fundus":
            output = random.choice(f_mr[f_abnormities.index(abnormity)])
            fOutput = torch.max(torch.tensor(output), fOutput)
    output = torch.concat([fOutput, oOutput]).to(device)
    return output

# randomly choose an image from a certain disease
# pass the image through all D1 models and return the concatenated result
def get_abnormity_nums_vector(device, disease, o_mr, f_mr, abnormity_num_models):
    setting = get_setting()
    disease_abnormity_num = setting.get_disease_abnormity_num(disease, "All Abnormities")
    
    abnormity_output = get_mr(device, disease, disease_abnormity_num, o_mr, f_mr)
    abnormity_output = abnormity_output.unsqueeze(0).type(torch.float32).to(device)
    abnormity_nums_vector = torch.empty([0]).to(device)
    for disease in abnormity_num_models.keys():
        temp_model = abnormity_num_models[disease]
        temp_model.eval()
        abnormity_num_output = temp_model(abnormity_output)[0]
        abnormity_num_output = torch.nn.functional.softmax(abnormity_num_output, 0)
        abnormity_nums_vector = torch.cat([abnormity_nums_vector, abnormity_num_output])
    return abnormity_nums_vector