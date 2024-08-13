import os
import random
import shutil
from PIL import Image
import torchvision.transforms as transforms

def copy_images(image_list, src_folder, dest_folder):
    for img_name in image_list:
        shutil.copy(os.path.join(src_folder, img_name), os.path.join(dest_folder, img_name))

def save_image(image, img_name, folder):
    image.save(os.path.join(folder, img_name))

def filter_images(images):
    ignore_suffixes = ['_fake.jpg', '_fake.jpeg', '_fake.JPG', '_fake.JPEG', '_fake.png', '_fake.PNG']
    return [img for img in images if not any(img.endswith(suffix) for suffix in ignore_suffixes)]

def flip_and_save(images, src_folder, dest_folder):
    flip_transform = transforms.RandomHorizontalFlip(p=1.0)
    for img_name in images:
        img_path = os.path.join(src_folder, img_name)
        flipped_image = flip_transform(Image.open(img_path))
        save_image(flipped_image, f"flipped_{img_name}", dest_folder)

def get_test_for_class(parent_A, parent_B, class_name, m):
    folder_A = f'{parent_A}{class_name}'
    folder_B = f'{parent_B}{class_name}'
    all_images = filter_images(os.listdir(folder_A))
    n = len(all_images)
    os.makedirs(folder_B, exist_ok=True)

    if n > m:
        selected_images = random.sample(all_images, m)
        copy_images(selected_images, folder_A, folder_B)
    elif n <= m <= 2 * n:
        copy_images(all_images, folder_A, folder_B)
        flip_and_save(random.sample(all_images, m - n), folder_A, folder_B)
    else:
        copy_images(all_images, folder_A, folder_B)
        flip_and_save(all_images, folder_A, folder_B)
        
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.8, 1.2))
        ])
        
        transformed_images = []
        for img_name in all_images:
            for i in range(int((m - 2 * n) / n) + 1):
                transformed_image = transform(Image.open(os.path.join(folder_A, img_name)).convert("RGB"))
                transformed_img_name = f"transformed_{i}_{img_name}"
                save_image(transformed_image, transformed_img_name, folder_B)
                transformed_images.append(transformed_img_name)
        
        if len(os.listdir(folder_B)) > m:
            for img_name in random.sample(transformed_images, len(os.listdir(folder_B)) - m):
                os.remove(os.path.join(folder_B, img_name))

    print(f"Operation completed. {folder_B} now contains {len(os.listdir(folder_B))} images.")




m = 500  # Desired number of images in folder_B
parent_A = "ODADS/Data/Data/Original/OCT/"
parent_B = "ODADS/Data/Data/Test/OCT/"
os.makedirs(parent_B, exist_ok=True)
all_classes = [f.name for f in os.scandir(parent_A) if f.is_dir()]
for class_name in all_classes:
    get_test_for_class(parent_A, parent_B, class_name, m)











"""
parent_folder_A = 'ODADS/Data/Data/Transformed/OCT'
parent_folder_B = 'ODADS/Data/Data/Test/OCT'
num = 5000

def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

for subfolder_A in os.listdir(parent_folder_A):
    path_A = os.path.join(parent_folder_A, subfolder_A)
    
    if os.path.isdir(path_A):
        subfolder_B = subfolder_A
        path_B = os.path.join(parent_folder_B, subfolder_B)
        ensure_folder_exists(path_B)

        all_files = [f for f in os.listdir(path_A) if os.path.isfile(os.path.join(path_A, f))]
        num_files_to_move = int(0.1 * num)
        files_to_move = random.sample(all_files, num_files_to_move)
        
        for file_name in files_to_move:
            source_file = os.path.join(path_A, file_name)
            destination_file = os.path.join(path_B, file_name)
            shutil.move(source_file, destination_file)
        print(f"Moved {num_files_to_move} files from '{subfolder_A}' to '{subfolder_B}'")

print("File moving process completed.")
"""