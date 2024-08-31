import os
import torch
from torchvision import transforms
from Abnormity_Models import abnormity_models
from Utils import utils
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def get_abnormities_probs(img, model, device):
    model.eval()
    img = utils.resize_long_edge(img)
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    output = model(img.to(device))
    output = torch.nn.functional.softmax(output[0], dim=0)
    # print(output)
    return output