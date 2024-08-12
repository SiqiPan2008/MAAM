import os
import shutil
import re

def copy_with_regex(pattern, src_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    num_files = 0
    num_skips = 0

    for root, _, files in os.walk(src_dir):
        for filename in files:
            if pattern.search(filename):
                src_file = os.path.join(root, filename)
                dest_file = os.path.join(dest_dir, filename)
                
                if os.path.exists(dest_file):
                    print(f"Skipped (exists): {dest_file}")
                    num_skips += 1
                else:
                    shutil.copy(src_file, dest_file)
                    print(f"Copied: {src_file} to {dest_file}")
                    num_files += 1
    
    print(f"moved {num_files} files to {dest_dir}")
    print(f"skipped {num_skips} files")

if __name__ == "__main__":
    pattern = re.compile(r'_fake\.(jpg|jpeg|png)$', re.IGNORECASE)
    src_directory = "results/Final_Neo_ERM/Transformed_Neo_ERM/maps_cyclegan/test_latest/images"
    dest_directory = "GAN-imgs/ERM"

    copy_with_regex(pattern, src_directory, dest_directory)