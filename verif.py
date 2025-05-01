import os
from pathlib import Path

# Define the dataset root
dataset_root = 'C:/Users/hp/Downloads/horse10/horse10'
labeled_data_dir = f'{dataset_root}/labeled-data'
data_dir = f'{dataset_root}/data'
image_extensions = ['.jpg', '.jpeg', '.png']  # Common image extensions

# Function to count images in a directory and its subdirectories
def count_images(directory):
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return 0
    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                count += 1
    return count

# Count images in labeled-data (original dataset)
labeled_data_count = count_images(labeled_data_dir)
print(f"Total images in labeled-data: {labeled_data_count}")

# Count images in data (processed train/val/test splits)
data_count = count_images(data_dir)
print(f"Total images in data (train/val/test): {data_count}")

# Count images in each split (if data directory exists)
if os.path.exists(data_dir):
    for split in ['train', 'val', 'test']:
        split_dir = f'{data_dir}/{split}/images'
        split_count = count_images(split_dir)
        print(f"{split.capitalize()} set: {split_count} images")