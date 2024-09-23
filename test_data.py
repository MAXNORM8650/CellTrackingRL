import cv2
import os

image_dir = 'synthetic_data/masks'
image_paths = sorted([
    os.path.join(image_dir, fname)
    for fname in os.listdir(image_dir)
    if fname.endswith('.png') and fname.startswith('mask')
])

for path in image_paths:
    image = cv2.imread(path)
    if image is None:
        print(f"Failed to read image: {path}")
    else:
        print(f"Successfully read image: {path} with shape {image.shape}")
