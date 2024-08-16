import os
import random
import shutil
from PIL import Image
import torchvision.transforms as transforms

# randomly select images from each class in a folder and copy to another folder

def copy_images(image_list, src_folder, dest_folder):
    for img_name in image_list:
        shutil.copy(os.path.join(src_folder, img_name), os.path.join(dest_folder, img_name))

m = 300
source_folder = "ODADS/Data/Data/Train/Fundus/"
target_folder = "ODADS/Data/Data/Test/Fundus/"
os.makedirs(target_folder, exist_ok=True)
all_classes = [f.name for f in os.scandir(source_folder) if f.is_dir()]
for class_name in all_classes:
    class_source_folder = f'{source_folder}{class_name}'
    class_target_folder = f'{target_folder}{class_name}'
    all_images = os.listdir(class_source_folder)
    random.shuffle(all_images)
    all_images = all_images[:m]
    os.makedirs(class_target_folder, exist_ok=True)
    copy_images(all_images, class_source_folder, class_target_folder)
    print(class_name)