import os
import random
import shutil
from PIL import Image
import torchvision.transforms as transforms

def copy_images(image_list, src_folder, dest_folder):
    for img_name in image_list:
        shutil.copy(os.path.join(src_folder, img_name), os.path.join(dest_folder, img_name))

m = 300  # Desired number of images in folder_B
parent_train = "ODADS/Data/Data/Train/Fundus/"
parent_test = "ODADS/Data/Data/Test/Fundus/"
os.makedirs(parent_test, exist_ok=True)
all_classes = [f.name for f in os.scandir(parent_train) if f.is_dir()]
for class_name in all_classes:
    folder_train = f'{parent_train}{class_name}'
    folder_test = f'{parent_test}{class_name}'
    all_images = os.listdir(folder_train)
    random.shuffle(all_images)
    all_images = all_images[:m]
    os.makedirs(folder_test, exist_ok=True)
    copy_images(all_images, folder_train, folder_test)
    print(class_name)