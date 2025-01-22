"""Select a (reproducibly) random subset of the dataset for human experiments/testing purposes.
For large datasets, recommend running as build_dataset.py --distribute
"""

import os
import sys
import argparse
import tqdm
import shutil
import uuid
import random
import pandas as pd
from PIL import Image
from sklearn.utils import resample


def get_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, required=True, help="Destination location of dataset")

    parser.add_argument('--dirmap_recipe_path', type=str, default=None, help="Path to dirmap csv. Default is dirmap_recipe.csv inside args.data_path")

    parser.add_argument('--resize', default=False, action='store_true', help="resize images as needed")
    parser.add_argument('--size_x', type=int, default=224, help="target image width")
    parser.add_argument('--size_y', type=int, default=224, help="target image height")
    parser.add_argument('--image_format', type=str, default=None, help="target img format (e.g. png, jpg")

    parser.add_argument('--obfuscate', default=False, action='store_true',
                        help="Use random directory and file names so user can't guess class by page source inspect. **This does NOT work currently, and is probably overkill")
    parser.add_argument('--distribute', default=False, action='store_true',
                        help="Within each class dir, make subdirs to avoid too many images in one dir (inefficient)")
    parser.add_argument('--max_ims_per_dir', type=int, default=1000, help="For obfuscate & distribute")

    parser.add_argument('--class_names', type=str, nargs='+', default=None, help="List of class names to include in the sampled dataset. Use like --class_names zebra hippopotamus turtle")
    parser.add_argument('--class_names_inclusive', action='store_true', help="If set, include classes that contain any of the provided class names as substrings")

    parser.add_argument('--subset', default=False, action='store_true', help="only build a subset")
    parser.add_argument('--n_per_class_train', type=int, default=100, help="n_train for subsetting")
    parser.add_argument('--n_per_class_val', type=int, default=50, help="n_val for subsetting")
    parser.add_argument('--n_per_class_test', type=int, default=50, help="n_test for subsetting")

    parser.add_argument('--oversample_balance', default=False, action='store_true', help="Balance the classes by oversampling minority classes (with replacement)")

    parser.add_argument('--json', default=False, action='store_true', help="Store dirmap as .json in addition to .csv")

    return parser.parse_args(argv)


def is_valid_image_file(file_path):
  # Check file name extension
  valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
  if os.path.splitext(file_path)[1].lower() not in valid_extensions:
    print(f"Invalid image file extension \"{file_path}\". Skipping this file...")
  # Verify that image file is intact
  try:
    with Image.open(file_path) as img:
      img.verify()  # Verify if it's an image
      return True
  except (IOError, SyntaxError) as e:
    print(f"Invalid image file {file_path}: {e}")
    return False


def oversample_df(df):
  oversampled_df = pd.DataFrame()
  
  for split in df['split'].unique():
    split_df = df[df['split'] == split]
    
    # Find the maximum class count within the split
    max_count = split_df['class'].value_counts().max()
    
    oversampled_split_df = pd.DataFrame()
    
    # Oversample each class within the split
    for class_name, group in split_df.groupby('class'):
      if len(group) < max_count:
        # Resample to add more instances from the existing ones
        additional_samples = resample(group, replace=True, n_samples=max_count - len(group), random_state=42)
        
        # Rename the im_path for the additional samples
        additional_samples = additional_samples.reset_index(drop=True)
        for i in range(len(additional_samples)):
          original_path = additional_samples.at[i, 'im_path']
          base, ext = os.path.splitext(original_path)
          additional_samples.at[i, 'im_path'] = f"{base}__copy{i+1}{ext}"
        
        oversampled_group = pd.concat([group, additional_samples])
      else:
        oversampled_group = group
        
      oversampled_split_df = pd.concat([oversampled_split_df, oversampled_group])
    
    oversampled_df = pd.concat([oversampled_df, oversampled_split_df])
  
  return oversampled_df


def remove_copy_suffix(path):
    if "__copy" in path: 
        base, ext = os.path.splitext(path)
        base = base.split("__")[0]
        return f"{base}{ext}"
    else:
        return path


args = get_args(sys.argv[1:])

if args.dirmap_recipe_path is None:
    dirmap_recipe_path = os.path.join(args.data_path, "dirmap_recipe.csv")
else:
    dirmap_recipe_path = args.dirmap_recipe_path

data_df = pd.read_csv(dirmap_recipe_path)

if args.class_names:

    def class_name_matches(class_name, target_names, inclusive):
        if inclusive:
            return any(target.lower() in class_name.lower() for target in target_names)
        else:
            return class_name in target_names

    # Keep only items with a class name in the provided list
    print(args.class_names)
    print(len(data_df)) 
    data_df = data_df[data_df['class'].apply(lambda x: class_name_matches(x, args.class_names, args.class_names_inclusive))]
    print(len(data_df)) 
    
    # Create a mapping of old class names to new class numbers
    matched_classes = sorted(data_df['class'].unique())
    class_name_to_num = {name: i for i, name in enumerate(matched_classes)}
    data_df['orig_class_num'] = data_df['class_num']
    data_df['class_num'] = data_df['class'].map(class_name_to_num)

if args.oversample_balance: 
    data_df = oversample_df(data_df)

if args.subset:
    classes_splits_dfs = []
    for hclass_idx, hclass in enumerate(data_df["class_num"].unique().tolist()):
        for split_idx, split in enumerate(["train", "val", "test"]):
            sample_n = args.__dict__["n_per_class_" + split]
            if sample_n > 0:
                class_split_df = data_df[(data_df["split"] == split) & (data_df["class_num"] == hclass)]
                classes_splits_dfs.append(class_split_df.sample(n=sample_n, random_state=int(hclass_idx*100 + split_idx)))
    data_df = pd.concat(classes_splits_dfs).reset_index(drop=True)

    # Keep a copy of the original path, in case we want to match up the datasets later.
    data_df["orig_im_path"] = data_df["im_path"]

# if args.obfuscate:
#     # Number of class-level directories is at least high enough so that not much more than 500 images per dir.
#     # But also make sure there are at least double the number of folders as the number of classes.
#
#     if args.obfuscate:
#     num_clevel_dirs = max(math.ceil(len(data_df) / 500), 2 * len(data_df["class_num"].unique()))
#
#     dist_dirs = ["d" + str(num) for num in range(num_clevel_dirs)]
# else:
#     clevel_dirs = list(data_df["class"].unique())


if args.obfuscate:
    data_df = data_df.sample(frac=1).reset_index(drop=True)
else:
    data_df = data_df.sort_values(by=["class_num", "split", "im_path"])

random.seed(0)
dfs = []
for split in list(data_df["split"].unique()):
    print("Building the", split, "set...")
    split_df = data_df[data_df["split"] == split]
    d_num, current_dir_count = 0, 0
    last_class = "null_class_definitely_not_a_real_class"
    for index, row in tqdm.tqdm(split_df.iterrows(), total=split_df.shape[0]):

        # Check that the image file is valid. If not, remove it. 
        if not is_valid_image_file(remove_copy_suffix(row["im_path"])):
            split_df = split_df.drop(index)
            continue

        # If we're starting to process images from a new class (in non-obfuscate, distribute mode) reset dir counter.
        if args.distribute and (not args.obfuscate) and (not row["class"] == last_class):
            d_num, current_dir_count = 0, 0
        last_class = row["class"]

        # If there are too many images in the dir currently being populated, switch to a new directory.
        if current_dir_count >= args.max_ims_per_dir:
            d_num += 1
            current_dir_count = 0
        current_dir_count += 1

        if args.obfuscate:
            ext = "." + os.path.basename(row["im_path"].split(".")[1])
            im_file_name = str(uuid.uuid5(uuid.NAMESPACE_DNS, os.path.basename(row["im_path"]))) + ext
            rel_im_path = os.path.join(row["split"], "d" + str(d_num), im_file_name)
        elif args.distribute:
            rel_im_path = os.path.join(row["split"], row["class"], "d" + str(d_num), os.path.basename(row["im_path"]))
        else:
            rel_im_path = os.path.join(row["split"], row["class"], os.path.basename(row["im_path"]))

        im_dest = os.path.join(args.data_path, rel_im_path)

        if not os.path.exists(os.path.dirname(im_dest)):
            os.makedirs(os.path.dirname(im_dest))

        # Copy the image directly if no resizing or reformatting needed, then exit this loop iteration with "continue"
        if args.image_format is None and not args.resize:
            shutil.copy2(remove_copy_suffix(row["im_path"]), im_dest)
            split_df.at[index, "im_path"] = rel_im_path
            continue  # Do not proceed to the resizing/reformatting step, start over with next loop iteration

        # If some resizing and/or reformatting needed:
        img = Image.open(remove_copy_suffix(row["im_path"]))

        width, height = img.size
        if not width == height:
            print("Warning, image not square with dims ", width, ", ", height, ":", rel_im_path)
        if not (width == args.size_x and height == args.size_y):
            img = img.resize((args.size_x, args.size_y))

        rel_im_path = os.path.join(
            os.path.dirname(rel_im_path),
            os.path.basename(rel_im_path).split(".")[0] + "." + args.image_format)
        img.save(os.path.join(args.data_path, rel_im_path), args.image_format.upper())
        split_df.at[index, "im_path"] = rel_im_path

    dfs.append(split_df)

data_df = pd.concat(dfs)
data_df = data_df.sort_values(by=["class_num", "split", "im_path"])
data_df.to_csv(os.path.join(args.data_path, "dirmap.csv"), index=False)
if args.json:
    data_df.to_json(os.path.join(args.data_path, "dirmap.json"), orient="records")
