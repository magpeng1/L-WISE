import os
import pandas as pd

original_df_path = "/media/KLAB37/morgan_data/mhist/annotations.csv"
data_df_dest = "/media/KLAB37/morgan_data/mhist/dirmap_recipe.csv"
class_num_dict = {"hp": 0, "ssa": 1}


df = pd.read_csv(original_df_path)

df["class"] = df["Majority Vote Label"].str.lower()

df["class_num"] = df.apply(lambda row: class_num_dict[row["class"]], axis=1)

df["im_path"] = df.apply(lambda row: os.path.join(os.path.dirname(original_df_path), "images", row["Image Name"]), axis=1)

df["split"] = df["Partition"]

df["class_num_uncertain"] = df["Number of Annotators who Selected SSA (Out of 7)"]/7 # range from 0 to 7, convert to [0, 1]

data_df = df[["split", "class", "class_num", "class_num_uncertain", "Number of Annotators who Selected SSA (Out of 7)", "im_path"]]

data_df = data_df.sort_values(by=["class_num", "split", "class_num_uncertain", "im_path"])

if not os.path.exists(os.path.dirname(data_df_dest)):
  os.makedirs(os.path.dirname(data_df_dest))

data_df.to_csv(data_df_dest, index=False)