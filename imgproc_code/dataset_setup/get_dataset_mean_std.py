import sys
from tqdm import tqdm
import argparse
import torch
import numpy as np
from torchvision import transforms
from robustness.datasets import *
from torch.utils.data import DataLoader
from welford import Welford
import random
import warnings
import math


def get_mean_std(loader, sample_frac=1.0):
    # Initialize a single Welford object for calculating mean and variance for RGB channels
    welford_instance = Welford()

    # Calculate the number of batches to process
    total_batches = len(loader)
    batches_to_process = math.ceil(total_batches * sample_frac)

    with tqdm(total=batches_to_process, unit="batch") as t_loader:
        for batch_idx, (images, _) in enumerate(loader):
            if batch_idx >= batches_to_process:
                break

            # Convert images to numpy and reshape to (batch_size*height*width, num_channels)
            # This format allows processing all pixels with their RGB values at once
            images_np = images.numpy().transpose(0, 2, 3, 1).reshape(-1, images.size(1))
            welford_instance.add_all(images_np)
            t_loader.update(1)

    # Extract mean and variance from the Welford instance
    mean = welford_instance.mean
    var = welford_instance.var_s  # Sample variance
    std = np.sqrt(var)  # Standard deviation is the square root of variance

    return mean, std


def get_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--n_workers', type=int, default=8, help="Number of dataloader worker threads")
    parser.add_argument('--dataset_path', type=str, default="/media/KLAB37/morgan_data/wbc/acevedo_wbc/robustness", help="Path to dataset")
    parser.add_argument('--dataset_name', type=str, default=None, help="OPTIONAL. Specify only if you want to use the Robustness version of the dataset (which comes with Caveats: it might do its own normalization). This is the name of dataset (see robustness/datasets.py)")
    parser.add_argument('--sample_frac', type=float, default=1.0, help="Randomly sample only a fraction of the dataset (good estimate in much less time)")

    return parser.parse_args(argv)


args = get_args(sys.argv[1:])

tfs = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
])

if args.dataset_name is None:  # Use ImageFolder
    dataset = datasets.ImageFolder(root=args.dataset_path, transform=tfs)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
else: # Use a Dataset already defined within the Robustness library
    warnings.warn("You might be loading the images with some normalization already from the dataset class inside the robustness library...")

    # The line below does something similar to dataset = WBC(args.dataset_path)   (for a dataset class called WBC in datasets.py)
    dataset = eval(f"{args.dataset_name}(\"{args.dataset_path}\")")

    train_loader, _ = dataset.make_loaders(batch_size=args.batch_size, val_batch_size=0, workers=args.n_workers, transforms=(tfs, tfs))

mean, std = get_mean_std(train_loader, sample_frac=args.sample_frac)

print("mean:", mean)
print("std:", std)
