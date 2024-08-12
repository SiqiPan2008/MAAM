import os
from PIL import Image

source_folder = './OCT-Original/Drusen_sel'
target_folder = './OCT-Original/Drusen'

os.makedirs(target_folder, exist_ok=True)

number = 0
for filename in os.listdir(source_folder):
    file_path = os.path.join(source_folder, filename)
    
    with Image.open(file_path) as img:  
        width, height = img.size
        aspect_ratio = width / height
        
        if 0.9 <= aspect_ratio <= 1.1:  
            top_prop = 0.06
            bottom_prop = 0.9
            horizontal_prop = 0.03
        else:
            top_prop = 0
            bottom_prop = 1
            horizontal_prop = 0
        top_crop = int(height * top_prop)
        bottom_crop = int(height * bottom_prop)
        left_crop = int(width * horizontal_prop)
        right_crop = width - left_crop
        cropped_img = img.crop((left_crop, top_crop, right_crop, bottom_crop))
        save_path = os.path.join(target_folder, filename)
        cropped_img.save(save_path)

    number += 1
    if number % 5 == 0:
        print(f"{number} imgs processed")

print("Processing complete")