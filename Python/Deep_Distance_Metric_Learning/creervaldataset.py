# This file was run to create a validation set 

import os
import shutil
import random

# Path to the main directory containing the MNIST images in ten subdirectories (0 to 9)
train_dir = '../mnist/train/'  # Replace with your path

# Path to the validation directory where you want to move the images
val_dir = '../mnist/validation/'

# Ensure the validation directory exists
os.makedirs(val_dir, exist_ok=True)

# Loop over each subdirectory (0 to 9)
for class_folder in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_folder)

    if os.path.isdir(class_path):  # Ensure it's a directory
        # Get a list of all files in the class directory
        files = os.listdir(class_path)

        # Calculate 10% of the images to move
        num_files_to_move = int(len(files) * 0.1)

        # Randomly select the files to move
        files_to_move = random.sample(files, num_files_to_move)

        # Create corresponding folder in validation directory
        val_class_path = os.path.join(val_dir, class_folder)
        os.makedirs(val_class_path, exist_ok=True)

        # Move the selected files to the validation folder
        for file_name in files_to_move:
            src_file = os.path.join(class_path, file_name)
            dest_file = os.path.join(val_class_path, file_name)
            shutil.move(src_file, dest_file)
            
        print(f"Moved {num_files_to_move} files from {class_folder} to validation.")

print("Finished moving 10% of the images to the validation folder.")

