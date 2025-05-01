import pandas as pd
import os
import cv2
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# Paths
dataset_root = 'C:/Users/hp/Downloads/horse10/horse10'
labeled_data_dir = f'{dataset_root}/labeled-data'
csv_path = f'{dataset_root}/labeled-data/CollectedData_Byron.csv'
output_dir = f'{dataset_root}/data'

# Create directories for YOLO dataset
for split in ['train', 'val', 'test']:
    os.makedirs(f'{output_dir}/{split}/images', exist_ok=True)
    os.makedirs(f'{output_dir}/{split}/labels', exist_ok=True)

# Load CSV and extract body parts
df_header = pd.read_csv(csv_path, nrows=1)
body_parts = df_header.iloc[0, 1::2].values  # Get body parts (x-coordinates)
num_keypoints = len(body_parts)  # Should be 22

# Create column names
columns = ['image_path']
for bp in body_parts:
    columns.extend([f'{bp}_x', f'{bp}_y'])

# Load CSV, skipping metadata rows
df = pd.read_csv(csv_path, skiprows=2)
df.columns = columns

# Get unique image paths
image_paths = df['image_path'].unique()

# Split dataset (80% train, 10% val, 10% test)
train_paths, temp_paths = train_test_split(image_paths, test_size=0.2, random_state=42)
val_paths, test_paths = train_test_split(temp_paths, test_size=0.5, random_state=42)

def save_annotations(image_paths, split):
    for img_path in image_paths:
        # Construct full image path
        full_img_path = os.path.join(labeled_data_dir, img_path)
        if not os.path.exists(full_img_path):
            print(f"Image not found: {full_img_path}")
            continue

        # Read image to get dimensions
        img = cv2.imread(full_img_path)
        if img is None:
            print(f"Failed to load image: {full_img_path}")
            continue
        h, w = img.shape[:2]

        # Get annotations for this image
        img_annotations = df[df['image_path'] == img_path]

        # Create label file
        label_path = f'{output_dir}/{split}/labels/{Path(img_path).stem}.txt'
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        with open(label_path, 'w') as f:
            for _, row in img_annotations.iterrows():
                keypoints = []
                x_coords = []
                y_coords = []

                # Process each keypoint
                for bp in body_parts:
                    x = row[f'{bp}_x']
                    y = row[f'{bp}_y']
                    if pd.notnull(x) and pd.notnull(y):
                        # Normalize coordinates
                        x_norm = x / w
                        y_norm = y / h
                        vis = 1  # Visible
                        x_coords.append(x)
                        y_coords.append(y)
                    else:
                        x_norm = 0
                        y_norm = 0
                        vis = 0  # Not visible
                    keypoints.extend([x_norm, y_norm, vis])

                # Compute bounding box from keypoints
                if x_coords and y_coords:
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    margin = 0.1  # Add 10% margin
                    w_bb = (x_max - x_min) * (1 + margin)
                    h_bb = (y_max - y_min) * (1 + margin)
                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    x_center_norm = x_center / w
                    y_center_norm = y_center / h
                    w_bb_norm = w_bb / w
                    h_bb_norm = h_bb / h
                else:
                    continue  # Skip if no valid keypoints

                # Write to file
                line = [0, x_center_norm, y_center_norm, w_bb_norm, h_bb_norm] + keypoints
                f.write(' '.join(map(str, line)) + '\n')

        # Copy image to dataset folder
        dest_img_path = f'{output_dir}/{split}/images/{Path(img_path).name}'
        os.makedirs(os.path.dirname(dest_img_path), exist_ok=True)
        shutil.copy(full_img_path, dest_img_path)

# Process splits
save_annotations(train_paths, 'train')
save_annotations(val_paths, 'val')
save_annotations(test_paths, 'test')

# Create data.yaml
data_yaml = f"""
train: ./data/train/images
val: ./data/val/images
test: ./data/test/images

nc: 1
names: ['horse']

kpt_shape: [{num_keypoints}, 3]  # {num_keypoints} keypoints, 3 for (x, y, visibility)
flip_idx: [{', '.join(map(str, range(num_keypoints)))}]  # Adjust if needed
"""
with open(f'{dataset_root}/data.yaml', 'w') as f:
    f.write(data_yaml)