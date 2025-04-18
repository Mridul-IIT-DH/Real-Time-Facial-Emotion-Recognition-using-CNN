import os
import cv2
import numpy as np
from utils import get_face
import tqdm # For progress bar

# Directories for training and testing data
data_dirs = {
    'train': './data/train',
    'test': './data/test'
}

# Target size for face images (grayscale)
image_size = (48, 48)

# Dictionary to hold data arrays
data = {}
emotion_labels_map = {} # To store mapping from index to label name

print("Starting data preparation...")

for split, data_dir in data_dirs.items():
    print(f"\nProcessing '{split}' data from: {data_dir}")
    X = []
    y = []
    emotion_folders = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    if not emotion_folders:
        print(f"Warning: No subdirectories found in {data_dir}. Skipping.")
        continue

    # Build label map only once using train directory structure
    if split == 'train':
        emotion_labels_map = {i: label for i, label in enumerate(emotion_folders)}
        print("Detected emotion labels:", emotion_labels_map)

    # Loop through each emotion folder (subfolder)
    for label_index, label in enumerate(emotion_folders):
        label_dir = os.path.join(data_dir, label)
        print(f"  Processing emotion: {label} (Index: {label_index})")

        # Process each image in the emotion folder with a progress bar
        image_files = os.listdir(label_dir)
        for file in tqdm.tqdm(image_files, desc=f"    {label}"):
            img_path = os.path.join(label_dir, file)

            # Read image (as color initially for face detection)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}. Skipping.")
                continue

            # Detect, crop, and get grayscale face
            face = get_face(img, size=image_size) # utils.get_face now returns grayscale
            if face is not None:
                X.append(face)
                y.append(label_index)  # Store label as index of emotion folder
            # else:
            #     print(f"Warning: No face detected in {img_path}. Skipping.")


    if not X:
         print(f"Warning: No valid faces processed for the '{split}' split. Check data or face detection.")
         continue

    # Convert to numpy arrays
    # --- IMPORTANT: Normalization/Scaling is REMOVED here ---
    # It will be done later by ImageDataGenerator or manually before model input
    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='int')

    # Add channel dimension for grayscale images (required by Conv2D)
    X = np.expand_dims(X, axis=-1)

    # Save arrays for this split
    output_dir = '.' # Save in the current directory
    os.makedirs(output_dir, exist_ok=True)
    x_filename = os.path.join(output_dir, f'X_{split}.npy')
    y_filename = os.path.join(output_dir, f'y_{split}.npy')

    np.save(x_filename, X)
    np.save(y_filename, y)

    # Print info about the saved dataset
    print(f"\nSaved '{split}' set: {len(X)} samples.")
    print(f"  Features saved to: {x_filename} (Shape: {X.shape})")
    print(f"  Labels saved to:   {y_filename} (Shape: {y.shape})")

# Save the label map
label_map_filename = os.path.join(output_dir, 'emotion_labels.npy')
np.save(label_map_filename, emotion_labels_map)
print(f"\nEmotion label map saved to: {label_map_filename}")

print("\nData preparation finished.")