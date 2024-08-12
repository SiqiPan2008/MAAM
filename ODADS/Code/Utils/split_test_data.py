import os
import shutil
import random

parent_folder_A = 'ODADS/Data/Data/Transformed/OCT'
parent_folder_B = 'ODADS/Data/Data/Test/OCT'
num = 5000

def ensure_subfolder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

for subfolder_A in os.listdir(parent_folder_A):
    path_A = os.path.join(parent_folder_A, subfolder_A)
    
    if os.path.isdir(path_A):
        subfolder_B = subfolder_A
        path_B = os.path.join(parent_folder_B, subfolder_B)
        ensure_subfolder_exists(path_B)

        all_files = [f for f in os.listdir(path_A) if os.path.isfile(os.path.join(path_A, f))]
        num_files_to_move = int(0.1 * num)
        files_to_move = random.sample(all_files, num_files_to_move)
        
        for file_name in files_to_move:
            source_file = os.path.join(path_A, file_name)
            destination_file = os.path.join(path_B, file_name)
            shutil.move(source_file, destination_file)
        print(f"Moved {num_files_to_move} files from '{subfolder_A}' to '{subfolder_B}'")

print("File moving process completed.")