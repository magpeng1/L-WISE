"""Define imagenet16 OR restricted_imagenet dataset.
IMPORTANT:
Before running this script, you must have run imagenet_df.py successfully.
"""

import os
import sys
import argparse
import pandas as pd


def get_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--subset', type=str, default='i16', help="Set to i16 for imagenet16 or ri for restricted imagenet")
    parser.add_argument('--imagenet_csv_path', type=str, default='data/imagenet_df.csv', help="Location of imagenet csv")
    parser.add_argument('--imagenet_path', type=str, default='/media/KLAB37/datasets/ImageNet2012', help="Location of imagenet")
    parser.add_argument('--dataset_dest_location', type=str, default='data/', help="Location of defined dataset")

    return parser.parse_args(argv)


args = get_args(sys.argv[1:])

if args.subset == "i16":
    dataset_dest_path = os.path.join(args.dataset_dest_location, "imagenet16")
elif args.subset == "ri":
    dataset_dest_path = os.path.join(args.dataset_dest_location, "restricted_imagenet")
else:
    raise ValueError("subset argument should be i16 or ri")

if not os.path.exists(dataset_dest_path):
    os.makedirs(dataset_dest_path)

imagenet_df = pd.read_csv(args.imagenet_csv_path)

imagenet_df["im_path"] = imagenet_df.apply(lambda row: os.path.join(args.imagenet_path, row["im_path"]), axis=1)

# Combine dataset dfs
all_datasets_df = imagenet_df

splits_dfs = []
for hclass in list(all_datasets_df[args.subset + "_superclass"].unique()):
    if hclass is not None:
      all_train_df = all_datasets_df[(all_datasets_df[args.subset + "_superclass"] == hclass) & (all_datasets_df["split"] == "train")]
      splits_dfs.append(all_train_df)

      all_val_df = all_datasets_df[(all_datasets_df[args.subset + "_superclass"] == hclass) & (all_datasets_df["split"] == "val")]
      splits_dfs.append(all_val_df)

assembled_df = pd.concat(splits_dfs)

assembled_df["orig_class"] = assembled_df["class"]
assembled_df["orig_class_num"] = assembled_df["class_num"].astype("int64")

assembled_df["class"] = assembled_df[args.subset + "_superclass"]
assembled_df["class_num"] = assembled_df[args.subset + "_superclass_num"]

assembled_df = assembled_df[["split", "class", "class_num", "orig_class", "orig_class_num", "im_path"]]
assembled_df = assembled_df.sort_values(by=["class_num", "orig_class_num", "split", "im_path"])
assembled_df["class_num"] = assembled_df["class_num"].astype("int64")

assembled_df.to_csv(os.path.join(dataset_dest_path, "dirmap_recipe.csv"), index=False)


print("Train set class breakdown:")
print(assembled_df[assembled_df["split"] == "train"]["class"].value_counts())

print("Val set class breakdown:")
print(assembled_df[assembled_df["split"] == "val"]["class"].value_counts())

print("Overall train-val split:")
print(assembled_df["split"].value_counts())
