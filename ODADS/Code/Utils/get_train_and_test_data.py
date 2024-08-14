import os
import random
import shutil
from PIL import Image
import torchvision.transforms as transforms

def resize_long_edge(img, longEdgeSize = 224):
    width, height = img.size
    if longEdgeSize == 0:
        longEdgeSize = max(width, height)
    if width > height:
        newSize = (longEdgeSize, int(height * longEdgeSize / width))
        loc = (0, int((longEdgeSize - newSize[1]) / 2))
    else:
        newSize = (int(longEdgeSize * width / height), longEdgeSize)
        width, _ = img.size
        loc = (int((longEdgeSize - newSize[0]) / 2), 0)
    img = img.resize(newSize)
    blackBackground = Image.new("RGB", (longEdgeSize, longEdgeSize), "black")
    blackBackground.paste(img, loc)
    return blackBackground

def copy_images(image_list, src_folder, dest_folder, resize, add_prefix = True):
    for img_name in image_list:
        img_path = os.path.join(src_folder, img_name)
        image = Image.open(img_path)
        if resize:
            image = resize_long_edge(image, longEdgeSize=resize)
        save_image(image, f"original_{img_name}" if add_prefix else img_name, dest_folder)

def save_image(image, img_name, folder):
    image.save(os.path.join(folder, img_name))

def filter_images(images):
    ignore_suffixes = ['_fake.jpg', '_fake.jpeg', '_fake.JPG', '_fake.JPEG', '_fake.png', '_fake.PNG']
    return [img for img in images if not any(img.endswith(suffix) for suffix in ignore_suffixes)]

def flip_and_save(images, src_folder, dest_folder, resize):
    flip_transform = transforms.RandomHorizontalFlip(p=1.0)
    for img_name in images:
        img_path = os.path.join(src_folder, img_name)
        flipped_image = flip_transform(Image.open(img_path))
        if resize:
            flipped_image = resize_long_edge(flipped_image, longEdgeSize=resize)
        save_image(flipped_image, f"flipped_{img_name}", dest_folder)

def move_flip_transform(folder_A, imgs, folder_B, dataType, resize, m, add_original_prefix=True):
    n = len(imgs)
    os.makedirs(folder_B, exist_ok=True)
    datatransforms = {
        "OCT Train": transforms.Compose([
            transforms.ColorJitter(brightness=0.1),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.8, 1.2)),
            transforms.Lambda(resize_long_edge)
        ]),
        "OCT Test": transforms.Compose([
            transforms.ColorJitter(brightness=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.8, 1.2))
        ]),
        "Fundus Train": transforms.Compose([
            transforms.ColorJitter(brightness=0.1),
            transforms.RandomApply([transforms.RandomRotation(30)], p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.8, 1.2)),
            transforms.Lambda(resize_long_edge)
        ]),
        "Fundus Test": transforms.Compose([
            transforms.ColorJitter(brightness=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.8, 1.2))
        ]),
    }

    if n >= m:
        selected_images = random.sample(imgs, m)
        copy_images(selected_images, folder_A, folder_B, resize = resize, add_prefix=add_original_prefix)
    elif n < m <= 2 * n:
        copy_images(imgs, folder_A, folder_B, resize = resize, add_prefix=add_original_prefix)
        flip_and_save(random.sample(imgs, m - n), folder_A, folder_B, resize = resize)
    else:
        copy_images(imgs, folder_A, folder_B, resize = resize, add_prefix=add_original_prefix)
        flip_and_save(imgs, folder_A, folder_B, resize = resize)
        
        transform = datatransforms[dataType]
        
        transformed_images = []
        for img_name in imgs:
            for i in range(int((m - 2 * n) / n) + 1):
                transformed_image = transform(Image.open(os.path.join(folder_A, img_name)).convert("RGB"))
                transformed_img_name = f"transformed_{i}_{img_name}"
                save_image(transformed_image, transformed_img_name, folder_B)
                transformed_images.append(transformed_img_name)
            print(f"{len(os.listdir(folder_B))}")
        
        if len(os.listdir(folder_B)) > m:
            for img_name in random.sample(transformed_images, len(os.listdir(folder_B)) - m):
                os.remove(os.path.join(folder_B, img_name))

    print(f"Operation completed. {folder_B} now contains {len(os.listdir(folder_B))} images.")
    

def get_temp_train_and_test(parent_A, class_name, train_folder_name, test_folder_name):
    folder_A = f'{parent_A}{class_name}'
    folder_temp_train = f'{train_folder_name}{class_name}'
    folder_test = f'{test_folder_name}{class_name}'
    all_images = [f for f in os.listdir(folder_A) if os.path.splitext(f)[1].lower() in [".png", ".jpg", ".jpeg"]]
    non_GAN_imgs = filter_images(all_images)
    GAN_imgs = [img for img in all_images if img not in non_GAN_imgs]
    
    random.shuffle(non_GAN_imgs)
    split_point = int(0.8 * len(non_GAN_imgs))
    train_non_GAN_imgs = non_GAN_imgs[:split_point]
    test_non_GAN_imgs = non_GAN_imgs[split_point:]
    
    train_imgs = train_non_GAN_imgs + GAN_imgs
    move_flip_transform(folder_A, train_imgs, folder_temp_train, "OCT Train", 269, len(train_imgs), add_original_prefix=False)
    move_flip_transform(folder_A, test_non_GAN_imgs, folder_test, "OCT Test", 0, 500)
    
def get_train(temp_train_folder_name, train_folder_name, class_name):
    folder_train = f'{train_folder_name}{class_name}'
    folder_temp_train = f'{temp_train_folder_name}{class_name}'
    all_images = [f for f in os.listdir(folder_temp_train) if os.path.splitext(f)[1].lower() in [".png", ".jpg", ".jpeg"]]
    move_flip_transform(folder_temp_train, all_images, folder_train, "OCT Train", 224, 5000)
    

parent_A = "ODADS/Data/Data/Original/OCT/"
temp_train_folder_name = "ODADS/Data/Data/Temp_train/OCT/"
train_folder_name = "ODADS/Data/Data/Train/OCT/"
test_folder_name = "ODADS/Data/Data/Test/OCT/"
os.makedirs(train_folder_name, exist_ok=True)
os.makedirs(temp_train_folder_name, exist_ok=True)
os.makedirs(test_folder_name, exist_ok=True)
all_classes = [f.name for f in os.scandir(parent_A) if f.is_dir()]
for class_name in all_classes:
    get_temp_train_and_test(parent_A, class_name, temp_train_folder_name, test_folder_name)
    get_train(temp_train_folder_name, train_folder_name, class_name)