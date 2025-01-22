import os
import pandas as pd
from sklearn.model_selection import train_test_split

val_split_prop = 0.3
data_dir = "/media/KLAB37/morgan_data/HAM10000"
data_df_dest = "/media/KLAB37/morgan_data/HAM10000/ham10000_df.csv"
task_df_dest = "/media/KLAB37/morgan_data/HAM10000/ham10000_df_VAL_4CLASSES.csv"

df = pd.read_csv(os.path.join(data_dir, "HAM10000_metadata"))

print("-------DATASET SOURCES-------")
print(df["dataset"].value_counts())

print("-------LOCALIZATIONS-------")
print(df["localization"].value_counts())

print("-------DIAGNOSIS TYPES-------")
print(df["dx_type"].value_counts())

df["class"] = df["dx"]
df["im_path"] = df.apply(lambda row: os.path.join(data_dir, "images", row["image_id"] + ".jpg"), axis=1)

# Add class_num column (integer index of class name in sorted list of class names)
unique_classes = sorted(df['class'].unique())
class_to_num = {cls: idx for idx, cls in enumerate(unique_classes)}
df['class_num'] = df['class'].map(class_to_num)

# Train/val split, making sure each lesion_id appears in only either train or val. 
unique_lesions = df['lesion_id'].unique()
train_lesions, val_lesions = train_test_split(unique_lesions, test_size=val_split_prop, random_state=42)
df['split'] = df['lesion_id'].apply(lambda x: 'train' if x in train_lesions else 'val')

df = df[["class_num", "class", "split", "im_path", "lesion_id", "dx_type", "age", "sex", "localization", "dataset"]]
df = df.sort_values(by=["class_num", "split", "lesion_id"])

# Calculate and print class distributions for train set
train_class_distribution = df[df['split'] == 'train']['class'].value_counts()
print("\nTrain set class distribution:\n", train_class_distribution)

# Calculate and print class distributions for val set
val_class_distribution = df[df['split'] == 'val']['class'].value_counts()
print("\nValidation set class distribution:\n", val_class_distribution)

overall_class_distribution = df['class'].value_counts()
print("\nOverall class distribution:\n", overall_class_distribution)

df.to_csv(data_df_dest, index=False)

df = df[df["class"].isin(["nv", "mel", "bkl", "bcc"])]
df = df[df["split"] == "val"]
df = df.drop_duplicates(subset='lesion_id')
df.to_csv(task_df_dest, index=False)