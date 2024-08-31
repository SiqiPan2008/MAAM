import os
import random
import shutil
from PIL import Image

# randomly select images from each class in a folder and copy to another folder
# alternatively, resize all images in each class in a folder to 224*224 and save to another folder  

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

def copy_images(image_list, src_folder, dest_folder):
    for img_name in image_list:
        shutil.copy(os.path.join(src_folder, img_name), os.path.join(dest_folder, img_name))

def resize_and_save_images(image_list, src_folder, dest_folder):
    for img_name in image_list:
        src_path = os.path.join(src_folder, img_name)
        dest_path = os.path.join(dest_folder, img_name)
        img = Image.open(src_path)
        img = resize_long_edge(img)
        img.save(dest_path)


resize_only = True
choose_and_copy = False
source_folder = "ODADS/Data/Images/TO_Original/Fundus/"
target_folder = "ODADS/Data/Images/TO/Fundus/"
os.makedirs(target_folder, exist_ok=True)
all_classes = [f.name for f in os.scandir(source_folder) if f.is_dir()]
for class_name in all_classes:
    class_source_folder = f'{source_folder}{class_name}'
    class_target_folder = f'{target_folder}{class_name}'
    all_images = os.listdir(class_source_folder)
    os.makedirs(class_target_folder, exist_ok=True)
    if resize_only:
        resize_and_save_images(all_images, class_source_folder, class_target_folder)
    elif choose_and_copy:
        m = 300
        random.shuffle(all_images)
        all_images = all_images[:m]
        copy_images(all_images, class_source_folder, class_target_folder)
    print(class_name)