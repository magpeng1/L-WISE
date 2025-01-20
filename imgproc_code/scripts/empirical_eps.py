import os
import torch
from PIL import Image
from torchvision.transforms import PILToTensor
import numpy as np
from tqdm import tqdm

def pil_to_tensor(img):
    return PILToTensor()(img)

def l2_distance(orig_img, image):
    # Convert to tensors and normalize
    orig_tensor = pil_to_tensor(orig_img).float() / 255
    image_tensor = pil_to_tensor(image).float() / 255
    
    # Ensure same dimensions
    assert orig_tensor.shape == image_tensor.shape, "Images must have the same dimensions"
    
    # Calculate L2 distance
    return torch.norm(orig_tensor - image_tensor).item()

def get_image_paths(directory):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def calculate_stats(distances):
    mean = np.mean(distances)
    std = np.std(distances)
    sem = std / np.sqrt(len(distances))
    return mean, std, sem

def main():
    base_dir = "data/for_experiments/imagenet16"
    directories = [
        "imagenet16_resized",
        "imagenet16_msrcr_resized",
        "imagenet16_clahe_2_resized",
        "imagenet16_auto_lr_resized",
        "imagenet16_20_0.5_40_logit"
    ]
    
    # Get image paths for the base directory
    base_image_paths = get_image_paths(os.path.join(base_dir, directories[0], "val"))
    print(f"Number of images in {directories[0]}: {len(base_image_paths)}")
    
    # Check and print image counts for all directories
    for directory in directories[1:]:
        image_paths = get_image_paths(os.path.join(base_dir, directory, "val"))
        print(f"Number of images in {directory}: {len(image_paths)}")
        assert len(image_paths) == len(base_image_paths), f"Image count mismatch in {directory}"
    
    # Calculate L2 distances
    results = {}
    for directory in directories[1:]:
        distances = []
        for img_path in tqdm(base_image_paths, desc=f"Processing {directory}"):
            relative_path = os.path.relpath(img_path, os.path.join(base_dir, directories[0], "val"))
            compare_path = os.path.join(base_dir, directory, "val", relative_path)
            
            orig_img = Image.open(img_path).convert('RGB')
            compare_img = Image.open(compare_path).convert('RGB')
            
            distance = l2_distance(orig_img, compare_img)
            distances.append(distance)
        
        mean, std, sem = calculate_stats(distances)
        results[directory] = {"mean": mean, "std": std, "sem": sem}
    
    # Print results
    print("\nResults:")
    for directory, stats in results.items():
        print(f"{directory}:")
        print(f"  Mean L2 distance: {stats['mean']:.4f}")
        print(f"  Standard deviation: {stats['std']:.4f}")
        print(f"  Standard error of the mean: {stats['sem']:.4f}")
        print()

if __name__ == "__main__":
    main()