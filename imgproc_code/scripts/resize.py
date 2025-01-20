"""Resize an entire dataset of images in a certain way. 
Run from root project folder using "python resize.py ..."
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent)) # Add parent directory to pythonpath
import argparse
import time
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop
from utils.dirmap_dataset import DirmapDataset, custom_collate


def get_args(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('--dest_dir', type=str, required=True, help="Location at which dest dir will be created")
    parser.add_argument('--dirmap_path', type=str, default=None, help="Path to dirmap csv.")
    parser.add_argument('--dataset_path', type=str, default="/media/KLAB37/datasets/ImageNet2012", help="Path to dataset")
    parser.add_argument('--num_workers', type=int, default=1, help="Number of CPU threads for dataloader")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")

    if argv is None:
      args = argparse.Namespace()  # Create a namespace with default values
      
      for action in parser._actions:  # Iterate over action objects and set default values to the namespace
        if action.default is not argparse.SUPPRESS:  # ignore the help action
          setattr(args, action.dest, action.default)
        
      args_dict = vars(args)
    else:
      args = parser.parse_args(argv)
      args_dict = vars(args)

    return args_dict


def resize_images(**kwargs):

  # Get defaults if this is being called as a function by an outside script
  if not __name__ == "__main__": 
    default_kwargs = get_args(None)
    default_kwargs.update(kwargs)
    kwargs = default_kwargs

  dest_dir = kwargs['dest_dir']

  print("Saving to:", dest_dir)

  if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

  transform = Compose([
    Resize(224),
    CenterCrop((224, 224)),
    ToTensor()
  ])

  class_num_col = "class_num"
  dataset = DirmapDataset(csv_file=kwargs['dirmap_path'], transform=transform, class_num_col=class_num_col, dataset_path=kwargs['dataset_path'], relative_path=True)
  val_loader = DataLoader(dataset, batch_size=kwargs['batch_size'], num_workers=kwargs['num_workers'], collate_fn=custom_collate)

  for data in val_loader:

    if kwargs['dirmap_path']:
      im, _, idx = data
    else:
      im, _ = data
      idx = None  # For predefined datasets, idx might not be used

    for j, img in enumerate(im):
      im_path = dataset.df.loc[idx[j], "im_path"]
      save_path = os.path.join(dest_dir, im_path)
      print(save_path)
      if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
      save_image(img, save_path)

    # Save the updated DataFrame
    new_csv_path = os.path.join(dest_dir, "dirmap.csv")
    dataset.df.to_csv(new_csv_path, index=False)


if __name__ == "__main__":
  start = time.time()
  args = get_args(sys.argv[1:])
  resize_images(**args)
  print("Runtime in seconds:", time.time() - start)