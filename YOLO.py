import os
import torch
import pandas as pd
import random
import shutil

def reduce_dataset(original_images_dir, original_labels_dir, new_images_dir, new_labels_dir, fraction=1/3):
    """
    Reduces the dataset by copying a fraction of the original images and labels.
    """

    # Ensure the new directories exist (train and val)
    os.makedirs(os.path.join(new_images_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(new_labels_dir, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(new_images_dir, 'valid', 'images'), exist_ok=True)
    os.makedirs(os.path.join(new_labels_dir, 'valid', 'labels'), exist_ok=True)


    # Process both train and valid datasets
    for subset in ['train', 'valid']:
        # Get all the image files for the subset (train or valid)
        image_files = [f for f in os.listdir(os.path.join(original_images_dir, subset, 'images')) if f.endswith('.jpg') or f.endswith('.png')]
        
        # Calculate how many images to keep
        num_images_to_keep = int(len(image_files) * fraction)
        print(f"Total {subset} images: {len(image_files)}, Reducing to: {num_images_to_keep}")
        
        # Randomly sample image files
        sampled_images = random.sample(image_files, num_images_to_keep)
        
        # Copy the sampled images and corresponding labels to the new directories
        for image in sampled_images:
            image_path = os.path.join(original_images_dir, subset, 'images', image)
            label_path = os.path.join(original_labels_dir, subset, 'labels', image.replace('.jpg', '.txt').replace('.png', '.txt'))
            
            # Create new paths
            new_image_path = os.path.join(new_images_dir, subset, 'images', image)
            new_label_path = os.path.join(new_labels_dir, subset, 'labels', image.replace('.jpg', '.txt').replace('.png', '.txt'))
            
            # Copy the image and label to the new directories
            shutil.copy(image_path, new_image_path)
            shutil.copy(label_path, new_label_path)


# Example usage
original_images_dir = '/Users/rohan/Desktop/yolov5'
original_labels_dir = '/Users/rohan/Desktop/yolov5'
new_images_dir = '/Users/rohan/Desktop/yolov5/reduced/images'
new_labels_dir = '/Users/rohan/Desktop/yolov5/reduced/labels'

# Reduce the dataset to 1/3 of its original size
reduce_dataset(original_images_dir, original_labels_dir, new_images_dir, new_labels_dir, fraction=1/3)

# Set paths
data_yaml_path = "/Users/rohan/Desktop/data.yaml"
weights = 'yolov5s.pt'  # Can also use yolov5m.pt, yolov5l.pt, or yolov5x.pt
img_size = 416  # Resized images from 640x640 to 416x416
batch_size = 8 # Reduced batch size to 8
epochs = 10 # Reduced epochs from 50 to 20
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Command to run YOLOv5 training in Python
os.system(f"python '/Users/rohan/Desktop/yolov5/train.py' --img {img_size} --batch {batch_size} --epochs {epochs} --data {data_yaml_path} --weights {weights} --device {device}")

# Path to the best weights file
best_weights = '/Users/rohan/Desktop/yolo5/runs/train/exp/weights/best.pt'

