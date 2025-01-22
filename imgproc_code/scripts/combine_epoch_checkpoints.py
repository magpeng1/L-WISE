"""
Neural Network Checkpoint Combiner

This script combines neural network epoch checkpoints from multiple directories into a single directory.
It renames the checkpoints to maintain a continuous sequence and handles special checkpoint files.

Functionality:
1. Combines checkpoints from multiple directories into one.
2. Renames checkpoints to maintain a continuous sequence across directories.
3. Ignores checkpoints with '_OBSOLETE' or any suffix after an underscore.
4. Copies 'checkpoint.pt.best' from a specified directory (or the last directory by default).
5. Always copies 'checkpoint.pt.latest' from the last directory.
6. Creates a 'origin_dirs_paths.txt' file listing the source directories and their epoch ranges.

Usage:
python checkpoint_combiner.py --dirs DIR1 DIR2 DIR3 ... --combined_dir_path COMBINED_DIR [--best_checkpoint_dir BEST_DIR]

Arguments:
  --dirs DIR1 DIR2 ...        List of directories containing checkpoints to combine (in order)
  --combined_dir_path DIR     Path to the new combined directory (will be created if it doesn't exist)
  --best_checkpoint_dir DIR   Optional: Directory to source 'checkpoint.pt.best' from (default: last directory)

Example:
python checkpoint_combiner.py --dirs /path/to/dir1 /path/to/dir2 /path/to/dir3 --combined_dir_path /path/to/combined_dir --best_checkpoint_dir /path/to/dir2

Notes:
- The script uses tqdm to display progress bars for each directory being processed.
- The combined directory will contain:
  - Renamed checkpoints (e.g., 0_checkpoint.pt, 1_checkpoint.pt, ...)
  - checkpoint.pt.best (from specified or last directory)
  - checkpoint.pt.latest (always from the last directory)
  - origin_dirs_paths.txt (listing source directories and their epoch ranges)

Requirements:
- Python 3.x
- tqdm library (install with 'pip install tqdm')
"""

import argparse
import os
import shutil
from tqdm import tqdm
import re

def get_args():
    parser = argparse.ArgumentParser(description="Combine neural network epoch checkpoints from multiple directories.")
    parser.add_argument('--dirs', nargs='+', required=True, help='List of directories to combine')
    parser.add_argument('--combined_dir_path', required=True, help='Path to the new combined directory')
    parser.add_argument('--best_checkpoint_dir', help='Directory to draw checkpoint.pt.best from (default: last directory)')
    return parser.parse_args()

def is_valid_checkpoint(filename):
    return filename.endswith('_checkpoint.pt') and '_' not in filename.split('_checkpoint.pt')[0]

def get_checkpoint_number(filename):
    return int(filename.split('_')[0])

def main():
    args = get_args()
    
    if not os.path.exists(args.combined_dir_path):
        os.makedirs(args.combined_dir_path)
    
    next_checkpoint_number = 0
    origin_dirs_info = []

    for dir_index, dir_path in enumerate(args.dirs):
        checkpoints = [f for f in os.listdir(dir_path) if is_valid_checkpoint(f)]
        checkpoints.sort(key=get_checkpoint_number)
        
        start_epoch = next_checkpoint_number
        
        for checkpoint in tqdm(checkpoints, desc=f"Processing directory {dir_index + 1}/{len(args.dirs)}"):
            old_path = os.path.join(dir_path, checkpoint)
            new_filename = f"{next_checkpoint_number}_checkpoint.pt"
            new_path = os.path.join(args.combined_dir_path, new_filename)
            shutil.copy2(old_path, new_path)
            next_checkpoint_number += 1
        
        end_epoch = next_checkpoint_number - 1
        origin_dirs_info.append(f"{dir_path}: epochs {start_epoch}-{end_epoch}")
    
    # Copy checkpoint.pt.best
    best_checkpoint_dir = args.best_checkpoint_dir or args.dirs[-1]
    best_src = os.path.join(best_checkpoint_dir, 'checkpoint.pt.best')
    if os.path.exists(best_src):
        best_dst = os.path.join(args.combined_dir_path, 'checkpoint.pt.best')
        shutil.copy2(best_src, best_dst)
    
    # Copy checkpoint.pt.latest from the last directory
    latest_src = os.path.join(args.dirs[-1], 'checkpoint.pt.latest')
    if os.path.exists(latest_src):
        latest_dst = os.path.join(args.combined_dir_path, 'checkpoint.pt.latest')
        shutil.copy2(latest_src, latest_dst)
    
    # Save origin_dirs_paths.txt
    with open(os.path.join(args.combined_dir_path, 'origin_dirs_paths.txt'), 'w') as f:
        for info in origin_dirs_info:
            f.write(f"{info}\n")

if __name__ == "__main__":
    main()