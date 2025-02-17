import pandas as pd
from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
import os


def custom_collate(batch):
    images, labels, idxs = zip(*batch)
    # Convert all images to have 3 channels if they don't already
    images = [img[:3, :, :] if img.size(0) == 4 else img.expand(3, -1, -1) if img.size(0) == 1 else img for img in images]
    images = torch.stack(images, 0)
    labels = torch.tensor(labels)
    return images, labels, idxs

class DirmapDataset(Dataset):
    def __init__(self, csv_file, transform=None, class_num_col='class_num', dataset_path=None, relative_path=False, split=None, mean=None, std=None, use_df_idx=False):
        if isinstance(csv_file, str):
            self.df = pd.read_csv(csv_file)
        elif isinstance(csv_file, pd.DataFrame):
            self.df = csv_file
        else:
            raise ValueError("Invalid input for 'csv_file' in DirmapDataset. Provide either a file path or a Pandas dataframe.")

        # Check if orig_im_path column exists, if not, create it from im_path
        if relative_path or 'orig_im_path' not in self.df.columns:
            self.df['orig_im_path'] = self.df['im_path']

        if split:
            self.df = self.df[self.df["split"] == split]
        
        # Add a transform if not provided
        if transform is None:
            self.transform = Compose([Resize((224, 224)), ToTensor()])
        else:
            self.transform = transform

        self.class_num_col = class_num_col

        self.dataset_path = dataset_path

        self.mean = mean
        self.std = std
        self.use_df_idx = use_df_idx


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['orig_im_path']
        if self.dataset_path and self.dataset_path not in row['orig_im_path']:
            image_path = os.path.join(self.dataset_path, row['orig_im_path'])
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        if image.size(0) == 4:
            print(f"Found 4-channel image at index {idx}, file: {image_path}")
            # Keep only the first 3 channels
            image = image[:3, :, :]
        elif image.size(0) == 1:
            # Expand grayscale to 3 channels
            image = image.expand(3, -1, -1)

        try:
            class_num = row[self.class_num_col]
        except KeyError:
            print("WARNING: no column in dataframe called", self.class_num_col, ", reverting to 'class_num'")
            self.class_num_col = 'class_num'
            class_num = row[self.class_num_col]

        if self.use_df_idx:
            idx_return = row.name
        else:
            idx_return = idx
        
        return image, int(class_num), idx_return  