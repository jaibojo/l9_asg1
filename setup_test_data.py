import os
from PIL import Image
import numpy as np

def create_test_dataset():
    # Create directories
    os.makedirs('test_data/train/class1', exist_ok=True)
    os.makedirs('test_data/train/class2', exist_ok=True)
    os.makedirs('test_data/val/class1', exist_ok=True)
    os.makedirs('test_data/val/class2', exist_ok=True)
    
    # Create random images for testing
    for split in ['train', 'val']:
        for class_name in ['class1', 'class2']:
            # Create 10 images for training, 5 for validation
            num_images = 10 if split == 'train' else 5
            for i in range(num_images):
                # Create a random RGB image
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(f'test_data/{split}/{class_name}/image_{i}.jpg')

if __name__ == '__main__':
    create_test_dataset()
    print("Test dataset created successfully!") 