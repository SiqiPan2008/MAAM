import torch
import numpy as np
from PIL import Image
from AbnormityModels import abnormityModel
from Utils import utils
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# from https://blog.csdn.net/sjhKevin/article/details/121747933

class CamExtractor():
    def __init__(self, model):
        self.model = model
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def forward_pass_on_convolutions(self, x):
        for param in self.model.parameters():
            param.requires_grad = True
        
        conv_output = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x.register_hook(self.save_gradient)
        conv_output.append(x)
        
        x = self.model.layer2(x)
        x.register_hook(self.save_gradient)
        conv_output.append(x)

        x = self.model.layer3(x)
        x.register_hook(self.save_gradient)
        conv_output.append(x)

        x = self.model.layer4(x)
        x.register_hook(self.save_gradient)
        conv_output.append(x)

        return conv_output, x
    
    def forward_pass(self, x):
        conv_output, x = self.forward_pass_on_convolutions(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        return conv_output, x





class GradCam():
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.extractor = CamExtractor(self.model)

    def generate_cam(self, input_image, target_layer, device, model, target_class=None):

        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class_idx = 0
            for idx in range(len(model_output[0])):
                if model_output[0][idx] > model_output[0][target_class_idx]:
                    target_class_idx = idx
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class_idx] = 1
        one_hot_output = one_hot_output.to(device)

        # Zero grads
        self.model.zero_grad()

        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)

        # Get hooked gradients,gradients
        guided_gradients = self.extractor.gradients[-1 - target_layer].data.cpu().numpy()[0]
        guided_gradients = torch.tensor(guided_gradients, dtype=torch.float32).to(device)

        # Get convolution outputs
        target = conv_output[target_layer].data.cpu().numpy()[0]
        target = torch.tensor(target, dtype=torch.float32).to(device)
        # Get weights from gradients
        weights = torch.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient

        # Create empty numpy array for cam
        cam = torch.ones(target.shape[1:], dtype=torch.float32).to(device)

        # Have a look at issue #11 to check why the above is np.ones and not np.zeros
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = torch.nn.functional.relu(cam)
        cam = (cam - torch.min(cam)) / (torch.max(cam) - torch.min(cam))  # Normalize between 0-1
        cam = (cam * 255).byte().cpu().numpy()  # Scale between 0-255 to visualize
        cam_resize = Image.fromarray(cam).resize((input_image.shape[2],
                                                  input_image.shape[3]), Image.LANCZOS)
        cam = np.uint8(cam_resize) / 255.0

        return cam

def camShow(img, cam, title = ""): 
    cam = np.array(cam)
    min = np.min(cam)
    max = np.max(cam)
    cam = (cam - min) / (max - min)
    
    fig, axs = plt.subplots(2, 3)
    #custom_cmap = "viridis"
    custom_cmap = [
        LinearSegmentedColormap.from_list('red_outline', [(0, 'black'), (0.3, 'black'), (0.5, 'red'), (0.7, 'black'), (1, 'black')]),
        LinearSegmentedColormap.from_list('red_yellow_green_blue', [(0, 'blue'), (0.4, 'green'), (0.6, 'yellow'), (1, 'red')]),
        LinearSegmentedColormap.from_list('red_blue_black', [(0, 'black'), (0.5, 'blue'), (1, 'red')])
    ]
    
    img = np.array(img).transpose((1, 2, 0))
    axs[0, 0].imshow(img)
    axs[0, 0].set_title(title + (" " if title else "") + "original img")
    
    map = axs[0, 1].imshow(cam)
    axs[0, 1].set_title(title + (" " if title else "") + "CAM")
    fig.colorbar(map, ax=axs[0, 1])

    black = np.zeros_like(img)
    alpha = cam[:, :, np.newaxis]
    filtered = img * alpha + black * (1 - alpha)
    axs[1, 0].imshow(filtered)
    axs[1, 0].set_title(title + (" " if title else "") + "overlaid alphamap")

    axs[1, 1].imshow(img)
    map0 = axs[1, 1].imshow(cam, cmap=custom_cmap[0], alpha=0.4)
    axs[1, 1].set_title(title + (" " if title else "") + "overlaid heatmap 0")
    fig.colorbar(map0, ax=axs[1, 1])
    
    axs[0, 2].imshow(img)
    map1 = axs[0, 2].imshow(cam, cmap=custom_cmap[1], alpha=0.4)
    axs[0, 2].set_title(title + (" " if title else "") + "overlaid heatmap 1")
    fig.colorbar(map1, ax=axs[0, 2])
    
    axs[1, 2].imshow(img)
    map2 = axs[1, 2].imshow(cam, cmap=custom_cmap[2], alpha=0.4)
    axs[1, 2].set_title(title + (" " if title else "") + "overlaid heatmap 2")
    fig.colorbar(map2, ax=axs[1, 2])

    axs[0, 0].axis("off")
    axs[0, 1].axis("off")
    axs[1, 0].axis("off")
    axs[1, 1].axis("off")
    plt.show()
    

def highlight(imgPath, numClasses, device, featureExtract, modelName, wtsName):
    model, _ = abnormityModel.initializeAbnormityModel(modelName, numClasses, featureExtract)
    model = model.to(device)
    trainedModel = torch.load(f"ODADS/Data/Weights/{wtsName}/{wtsName}.pth")
    model.load_state_dict(trainedModel["state_dict"])
    img = Image.open(imgPath)
    img = utils.processImg(img, customResize = 224)
    imgUnsqueezed = img.unsqueeze(0)
    imgUnsqueezed = imgUnsqueezed.to(device)
    modelWithGradCam = GradCam(model)
    cam = modelWithGradCam.generate_cam(imgUnsqueezed, 3, device, model)
    camShow(img, cam)