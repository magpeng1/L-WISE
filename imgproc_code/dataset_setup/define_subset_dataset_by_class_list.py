# Example usage: 
# python dataset_setup/define_subset_dataset_by_class_list.py --dataset_dest_location data/my_dataset_subset --class_indices 0 546 999 

import os
import sys
import argparse
import pandas as pd

def get_args(argv):
  parser = argparse.ArgumentParser()

  parser.add_argument('--class_indices', type=int, nargs='+', help="List of class indices to include")
  parser.add_argument('--dataset_csv_path', type=str, default='data/dataset_df.csv', help="Location of dataset csv")
  parser.add_argument('--dataset_path', type=str, default='/media/KLAB37/datasets/ImageNet2012', help="Location of dataset")
  parser.add_argument('--dataset_dest_location', type=str, default='data/custom_dataset_subset', help="Location of defined dataset")

  return parser.parse_args(argv)

args = get_args(sys.argv[1:])

if not os.path.exists(args.dataset_dest_location):
  os.makedirs(args.dataset_dest_location)

# Load the ImageNet CSV
dataset_df = pd.read_csv(args.dataset_csv_path)

# Update image paths to absolute paths
dataset_df["im_path"] = dataset_df.apply(lambda row: os.path.join(args.dataset_path, row["im_path"]), axis=1)

# Filter dataset based on provided class indices
filtered_df = dataset_df[dataset_df["class_num"].isin(args.class_indices)]

# Assign original class numbers to a new column
filtered_df["orig_class_num"] = filtered_df["class_num"]

# Reset class_num to a new numbering from 0 to number of classes - 1
class_mapping = {original: new for new, original in enumerate(sorted(args.class_indices))}
filtered_df["class_num"] = filtered_df["class_num"].map(class_mapping)

# Select and reorder the columns
filtered_df = filtered_df[["split", "class", "class_num", "orig_class_num", "im_path"]]
filtered_df = filtered_df.sort_values(by=["class_num", "split", "im_path"])

# Save the new dataset CSV
filtered_df.to_csv(os.path.join(args.dataset_dest_location, "dirmap_recipe.csv"), index=False)

print("Subset class breakdown:")
print(filtered_df["class"].value_counts())

print("Overall train-val split:")
print(filtered_df["split"].value_counts())
