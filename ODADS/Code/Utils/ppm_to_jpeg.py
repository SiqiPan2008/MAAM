import os
from PIL import Image

# convert all images within root_folder from .ppm to .jpg

def convert_ppm_to_jpg(root_folder):
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith('.ppm'):
                ppm_path = os.path.join(root, file)

                with Image.open(ppm_path) as img:
                    jpg_path = os.path.splitext(ppm_path)[0] + '.jpg'
                    img.save(jpg_path, 'JPEG')
                
                os.remove(ppm_path)
                print(f'Converted and deleted: {ppm_path}')

root_folder = ''
convert_ppm_to_jpg(root_folder)