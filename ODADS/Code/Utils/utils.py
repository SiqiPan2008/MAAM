from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import json
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
        
