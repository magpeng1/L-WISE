import os
import pandas as pd
import json
from tqdm import tqdm
from robustness.tools.constants import RESTRICTED_IMAGNET_RANGES as restricted_imagenet_ranges
from robustness.tools.constants import IMAGENET_16_RANGES as i16_ranges


data_dir = "/media/KLAB37/datasets/ImageNet2012"
data_df_dest = os.path.join("data", "imagenet_df.csv")

# Function to create both superclass_dict and updated id_dict with superclass information
def create_dictionaries_with_superclass(imagenet_classes, ri_ranges, ri_superclass_names, i16_ranges):
    ri_superclass_dict = {}
    id_dict = {}

    # Create initial restricted imagenet superclass_dict
    for range_pair, name in zip(ri_ranges, ri_superclass_names):
        start, end = range_pair
        class_ids = [imagenet_classes[str(i)][0] for i in range(start, end + 1) if str(i) in imagenet_classes]
        ri_superclass_dict[name] = class_ids

    # Create easy imagenet superclass dict
    i16_superclass_dict = {}
    for name in i16_ranges.keys():
        class_indices = i16_ranges[name]
        if isinstance(class_indices, tuple):
            start, end = class_indices
            class_idx_list = list(range(start, end + 1))
        elif isinstance(class_indices, list):
            class_idx_list = class_indices
        else:
            raise ValueError("Invalid superclass range entry:", class_indices, "of type", type(class_indices))
        class_ids = [imagenet_classes[str(i)][0] for i in class_idx_list if str(i) in imagenet_classes]
        i16_superclass_dict[name] = class_ids

    # Create id_dict and add superclass information
    for int_id, (alpha_id, class_name) in imagenet_classes.items():
        # Determine the superclass for each class
        ri_superclass = next((name for name, ids in ri_superclass_dict.items() if alpha_id in ids), None)
        i16_superclass = next((name for name, ids in i16_superclass_dict.items() if alpha_id in ids), None)
        id_dict[alpha_id] = {
            "int_id": int(int_id), 
            "class_name": class_name, 
            "ri_superclass_int_id": ri_superclass_names.index(ri_superclass) if ri_superclass is not None else None,
            "ri_superclass": ri_superclass,
            "i16_superclass_int_id": list(i16_ranges.keys()).index(i16_superclass) if i16_superclass is not None else None,
            "i16_superclass": i16_superclass,
        }

    return ri_superclass_dict, i16_superclass_dict, id_dict


# Source: https://github.com/raghakot/keras-vis/blob/master/resources/imagenet_class_index.json
with open(os.path.join(os.path.dirname(__file__), 'imagenet_class_index.json')) as f:
    imagenet_idx = json.load(f)

ri_superclass_names = ["dog", "cat", "frog", "turtle", "bird", "primate", "fish", "crab", "insect"]
ri_superclass_dict, i16_superclass_dict, id_dict = create_dictionaries_with_superclass(imagenet_idx, restricted_imagenet_ranges, ri_superclass_names, i16_ranges)

splits = ["train", "val"]

split_class_dfs = []
for split in splits:
    print("Processing images for:", split)
    class_ids = os.listdir(os.path.join(data_dir, split))
    for hclass in tqdm(class_ids):
        im_files = os.listdir(os.path.join(data_dir, split, hclass))
        hclass_df = pd.DataFrame(im_files, columns=["im_path"])
        hclass_df["im_path"] = os.path.join(split, hclass) + "/" + hclass_df["im_path"].astype(str)
        hclass_df["class"] = id_dict[hclass]["class_name"]
        hclass_df["class_num"] = id_dict[hclass]["int_id"]
        hclass_df["ri_superclass"] = id_dict[hclass]["ri_superclass"]
        hclass_df["ri_superclass_num"] = id_dict[hclass]["ri_superclass_int_id"]
        hclass_df["i16_superclass"] = id_dict[hclass]["i16_superclass"]
        hclass_df["i16_superclass_num"] = id_dict[hclass]["i16_superclass_int_id"]
        hclass_df["split"] = split
        split_class_dfs.append(hclass_df)

data_df = pd.concat(split_class_dfs, ignore_index=True)
data_df = data_df.sort_values(by=["class_num", "split", "im_path"])
data_df = data_df[["split", "class", "class_num", "ri_superclass", "ri_superclass_num", "i16_superclass", "i16_superclass_num", "im_path"]]

print("Class counts: ")
print(data_df["class"].value_counts())

print("RestrictedImagenet superclass counts: ")
print(data_df["ri_superclass"].value_counts())

print("ImageNet16 superclass counts: ")
print(data_df["i16_superclass"].value_counts())

data_df.to_csv(data_df_dest, index=False)
