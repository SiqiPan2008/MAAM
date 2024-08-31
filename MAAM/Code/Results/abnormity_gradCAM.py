import torch
import numpy as np
from PIL import Image
from Abnormity_Models import abnormity_models
from Utils import utils
from torchvision import transforms
import os
from Abnormity_Models import classify_abnormity
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# from https://blog.csdn.net/sjhKevin/article/details/121747933

custom_cmap = [
        LinearSegmentedColormap.from_list('red_outline', [(0, 'black'), (0.6, 'black'), (0.7, 'red'), (0.8, 'black'), (1, 'black')]),
        LinearSegmentedColormap.from_list('red_yellow_green_blue', [(0, 'blue'), (0.4, 'green'), (0.6, 'yellow'), (0.8, 'orange'), (1, 'red')]),
        LinearSegmentedColormap.from_list('red_blue_black', [(0, 'black'), (0.5, 'blue'), (1, 'red')])
    ]

class Cam_Extractor():
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

class GradCAM():
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.extractor = Cam_Extractor(self.model)

    def generate_cam(self, input_image, target_layer, device, model, target_class_idx=None, img_size = (224, 224)):

        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class_idx is None:
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
        cam_array = torch.ones(target.shape[1:], dtype=torch.float32).to(device)

        # Have a look at issue #11 to check why the above is np.ones and not np.zeros
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam_array += w * target[i, :, :]
        cam_array = torch.nn.functional.relu(cam_array)
        cam_array = (cam_array - torch.min(cam_array)) / (torch.max(cam_array) - torch.min(cam_array))  # Normalize between 0-1
        cam_array = (cam_array * 255).byte().cpu().numpy()  # Scale between 0-255 to visualize

        return cam_array

def cam_show_img(img, cam, title = ""): 
    cam = np.array(cam)
    min = np.min(cam)
    max = np.max(cam)
    cam = (cam - min) / (max - min)
    
    fig, axs = plt.subplots(2, 3)
    
    
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
    map0 = axs[1, 1].imshow(cam, cmap=custom_cmap[0], alpha=0.5)
    axs[1, 1].set_title(title + (" " if title else "") + "overlaid heatmap 0")
    fig.colorbar(map0, ax=axs[1, 1])
    
    axs[0, 2].imshow(img)
    map1 = axs[0, 2].imshow(cam, cmap=custom_cmap[1], alpha=0.4)
    axs[0, 2].set_title(title + (" " if title else "") + "overlaid heatmap 1")
    fig.colorbar(map1, ax=axs[0, 2])
    
    axs[1, 2].imshow(img)
    map2 = axs[1, 2].imshow(cam, alpha=0.5) # cmap=custom_cmap[2], 
    axs[1, 2].set_title(title + (" " if title else "") + "overlaid heatmap 2")
    fig.colorbar(map2, ax=axs[1, 2])

    axs[0, 0].axis("off")
    axs[0, 1].axis("off")
    axs[1, 0].axis("off")
    axs[1, 1].axis("off")
    plt.show()

def save_cam_single_img(img, cam):
    setting = utils.get_setting()
    fig_folder = setting.fig_folder
    fig_file = "GradCAM/0.png"
    fig_path = os.path.join(fig_folder, fig_file)
    
    cam = np.array(cam)
    min = np.min(cam)
    max = np.max(cam)
    cam = (cam - min) / (max - min)
    
    img = np.array(img).transpose((1, 2, 0))
    plt.imshow(img)
    plt.imshow(cam, alpha=0.5, cmap = custom_cmap[1])

    plt.axis("off")
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0)
    #plt.show()

def highlight(device, img_name, target_class = None):
    setting = utils.get_setting()
    img_path = os.path.join(setting.images_folder, img_name)
    type = "OCT" if setting.img_path_is_OCT(img_path) else "Fundus"
    A_folder = setting.A_folder
    best_net = setting.best_o_net if type == "OCT" else setting.best_f_net
    wts_name = best_net + "AOTRWT.pth" if type == "OCT" else best_net + "AFTRWT.pth"
    wts_path = os.path.join(A_folder, wts_name)
    num_classes = setting.get_abnormity_num("OCT Abnormities") if type == "OCT" else setting.get_abnormity_num("Fundus Abnormities")
    abnormities = setting.get_abnormities("OCT Abnormities") if type == "OCT" else setting.get_abnormities("Fundus Abnormities")
    
    model = abnormity_models.initialize_abnormity_model(best_net, num_classes, True)
    model = model.to(device)
    trained_model = torch.load(wts_path)
    model.load_state_dict(trained_model["state_dict"])
    img = Image.open(img_path)
    
    output = classify_abnormity.get_abnormities_probs(img, model, device)
    print(output)
    pred = torch.argmax(output)
    print(abnormities[pred][1])
    top_prob_indices = utils.get_top_prob_indices(output, setting.A_top_probs_max_num, setting.A_top_probs_min_prob)
    print([abnormities[idx][1] for idx in top_prob_indices])
    
    img_resized = utils.resize_and_to_tensor(img, custom_resize = 224)
    img = transforms.ToTensor()(img)
    img_unsqueezed = img_resized.unsqueeze(0)
    img_unsqueezed = img_unsqueezed.to(device)
    model_with_grad_cam = GradCAM(model)
    target_class_idx = None
    if not target_class is None:
        for i in range(num_classes):
            if abnormities[i][1] == target_class:
                target_class_idx = i
        
    cam_array = model_with_grad_cam.generate_cam(img_unsqueezed, 3, device, model, target_class_idx)
    
    #cam_224 = utils.resize_array_to_img_size(cam_array, (224, 224))
    #cam_show_img(img_resized, cam_224)

    cam_with_original_size = utils.resize_array_to_img_size(cam_array, [img.shape[-2], img.shape[-1]])
    save_cam_single_img(img, cam_with_original_size)