import os
from PIL import Image

def convert_ppm_to_jpg(root_folder):
    # 遍历文件夹及其子目录
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith('.ppm'):
                # 构建文件路径
                ppm_path = os.path.join(root, file)
                
                # 打开ppm文件
                with Image.open(ppm_path) as img:
                    # 构建jpg文件路径
                    jpg_path = os.path.splitext(ppm_path)[0] + '.jpg'
                    
                    # 保存为jpg文件
                    img.save(jpg_path, 'JPEG')
                
                # 删除原始ppm文件
                os.remove(ppm_path)
                print(f'Converted and deleted: {ppm_path}')

# 使用示例
root_folder = './Fundus-Original - Copy'
convert_ppm_to_jpg(root_folder)