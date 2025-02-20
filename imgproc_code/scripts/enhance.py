"""Enhance an entire dataset of images using a neural model. 
Run from imgproc_code folder using "python scripts/enhance.py ..."
"""

import os
import sys
import argparse
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import imageio
import torch as ch
import time
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
import torchvision.transforms.functional as TF
from robustness.datasets import *
from robustness.model_utils import make_and_restore_model
from lwise_imgproc_utils.dirmap_dataset import DirmapDataset, custom_collate
from lwise_imgproc_utils.mapped_models import imagenet_mapped_model
from lwise_imgproc_utils.custom_losses import *
from lwise_imgproc_utils.gif_utils import *
from robustness.tools.constants import IMAGENET_16_RANGES as i16_ranges
from robustness import data_augmentation
import timm
import dill
from robustness.imagenet_models import *


def get_args(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('--dest_dir', type=str, required=True, help="Location at which dest dir will be created")
    parser.add_argument('--dirmap_path', type=str, default=None, help="Path to dirmap csv.")
    parser.add_argument('--dataset_path', type=str, default="/media/KLAB37/datasets/ImageNet2012", help="Path to dataset")
    parser.add_argument('--dataset_name', type=str, default=None, required=False, help="Name of dataset as defined in Robustness library (see robustness/datasets.py). Or, one of the superclass subset names (i.e. imagenet16). Can leave unspecified if using superclasses.")
    parser.add_argument('--model_ckpt_path', type=str, default="/home/morgan/projects/learn-histopath-backend/model_ckpts/imagenet_l2_3_0.pt", help="Path to model checkpoint")
    parser.add_argument('--arch', type=str, default='resnet50', help="Name of CNN archhitecture (e.g. resnet50, convnext_tiny, densenet201)")
    
    # Setting up the optimization
    parser.add_argument('--eps', default=10, type=float, help="Epsilon budget for perturbation")
    parser.add_argument('--step_size', default=0.5, type=float, help="Perturbation step size")
    parser.add_argument('--num_steps', default=20, type=int, help="Number of steps for perturbation")
    parser.add_argument('--objective_type', type=str, default="logit", help="Type of objective for enhancement. cross_entropy | logit | logit_diverge")
    parser.add_argument('--diverge_from', nargs='+', help="Specific classes to diverge from, if using --objective_type logit_diverge. E.g., --diverge_from class1 class2 class3. All non-groundtruth classes are diverged from by default")
    parser.add_argument('--superclass', type=str, default=None, help="Optimize toward superclass label instead of fine-grained class label. restrictedimagenet | imagenet16")
    parser.add_argument('--attack', default=False, action='store_true', help="Do an attack instead of enhancement")

    parser.add_argument('--save_originals', default=False, action='store_true', help="Save original images")

    parser.add_argument('--num_workers', type=int, default=1, help="Number of CPU threads for dataloader")
    parser.add_argument('--gpu_id', type=int, default=0, help="ID of GPU to use")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")

    # Create GIFs of the enhancement (experimental)
    parser.add_argument('--make_gifs', default=False, action='store_true', help="Make a GIF of each enhancement")
    parser.add_argument('--gif_forward_backward_loop', default=False, action='store_true', help="GIFs move in backwards and forwards directions. ")
    parser.add_argument('--gif_border', default=False, action='store_true', help="Make a color-changing border for the gif.")
    parser.add_argument('--gif_extremes', default=False, action='store_true', help="Store only the extremes of the image. If using this flag, best to set --gif_fps 1")
    parser.add_argument('--gif_fps', type=int, default=15, help="Frames per second for gif generation")
    
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


def enhance_images(**kwargs):

  # Get defaults if this is being called as a function by an outside script
  if not __name__ == "__main__": 
    default_kwargs = get_args(None)
    default_kwargs.update(kwargs)
    kwargs = default_kwargs

  eps_fname = int(kwargs['eps']) if float(kwargs['eps']).is_integer() else kwargs['eps'] # Integer in file name if appropriate
  step_size_fname = int(kwargs['step_size']) if float(kwargs['step_size']).is_integer() else kwargs['step_size']

  perturb_id = str(eps_fname) + "_" + str(step_size_fname) + "_" + str(kwargs['num_steps']) + "_" + kwargs['objective_type']

  if kwargs['diverge_from'] and len(kwargs['diverge_from']) > 0:
    perturb_id = perturb_id + "_from_" + '_'.join(kwargs['diverge_from'])

  if kwargs['superclass']:
    perturb_id = perturb_id + "_" + kwargs['superclass']
  if kwargs['attack']:
    perturb_id = perturb_id + "_attack"

  if kwargs['dirmap_path']:
    dest_dir_name = os.path.basename(os.path.dirname(kwargs['dirmap_path'])) + "_" + perturb_id
    dirmap = pd.read_csv(kwargs['dirmap_path'])
    if kwargs['diverge_from'] and len(kwargs['diverge_from']) > 0: # Remove any instances of classes we are diverging from
      dirmap = dirmap[~dirmap['class'].isin(kwargs['diverge_from'])].reset_index()
  else:
    dest_dir_name = perturb_id

  dest_dir = os.path.join(kwargs['dest_dir'], dest_dir_name)

  print("Saving to:", dest_dir)

  if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

  if kwargs['save_originals']:
    if kwargs['dirmap_path']:
      orig_dest_dir_name = os.path.basename(os.path.dirname(kwargs['dirmap_path'])) + "_natural"
    else:
      orig_dest_dir_name = "natural"

    orig_dest_dir = os.path.join(kwargs['dest_dir'], orig_dest_dir_name)
    
    if not os.path.exists(orig_dest_dir):
      os.makedirs(orig_dest_dir)

  if kwargs['superclass']:
    if kwargs['superclass'].lower() in ["imagenet16"]:
      rb_ds = CustomImageNet(kwargs['dataset_path'], list(i16_ranges.values()))
    else:
      raise ValueError("Undefined superclass setting \"" + kwargs['superclass'] + "\"")
  else:
    rb_ds = eval(f"{kwargs['dataset_name']}(\"{kwargs['dataset_path']}\")")

  if kwargs['dirmap_path']:
    if "cifar" in kwargs['dataset_name'].lower():
      transform = data_augmentation.TEST_TRANSFORMS_DEFAULT(32)
    else:
      transform = Compose([
        transforms.Resize(224),
        transforms.CenterCrop((224, 224)),
        ToTensor()
      ])
    if kwargs['dataset_name'] in ["ImageNet", "inat"]:
      class_num_col = "class_num" if kwargs['superclass'] else "orig_class_num"
    else:
      class_num_col = "class_num"
    print("Using this column of provided dirmap for class indices: " + class_num_col)
    dataset = DirmapDataset(csv_file=dirmap, transform=transform, class_num_col=class_num_col, dataset_path=kwargs['dataset_path'])
    val_loader = DataLoader(dataset, batch_size=kwargs['batch_size'], num_workers=kwargs['num_workers'], collate_fn=custom_collate)
  else:
    _, val_loader = rb_ds.make_loaders(workers=kwargs['num_workers'], batch_size=kwargs['batch_size'])

  if kwargs['arch'] == "deb_convnext_tiny":
    convnext = timm.models.convnext.convnext_tiny(pretrained=False)
    sd = ch.load(kwargs['model_ckpt_path'], pickle_module=dill)
    convnext.load_state_dict(sd)
    model, _ = make_and_restore_model(arch=convnext, dataset=rb_ds, gpu_id=kwargs['gpu_id'])
  elif 'xcit' in kwargs['arch']:
    xcit_model = eval(kwargs['arch'] + "(pretrained=False)")
    state_dict = ch.load(kwargs['model_ckpt_path'], pickle_module=dill)
    xcit_model.load_state_dict(state_dict)
    model, _ = make_and_restore_model(arch=xcit_model, dataset=rb_ds, gpu_id=kwargs['gpu_id'])
  else:
    if kwargs['superclass']:
      model, _ = imagenet_mapped_model(arch=kwargs['arch'], superclassed_imagenet_ds=rb_ds, pytorch_pretrained=False, gpu_id=kwargs['gpu_id'], resume_path=kwargs['model_ckpt_path'])
    else:
      try:
        model, _ = make_and_restore_model(arch=kwargs['arch'], dataset=rb_ds, gpu_id=kwargs['gpu_id'], resume_path=kwargs['model_ckpt_path'])
      except RuntimeError:
        model, _ = make_and_restore_model(arch=kwargs['arch'], dataset=rb_ds, gpu_id=kwargs['gpu_id'])
        try:
            model, _ = make_and_restore_model(arch=model.model, dataset=rb_ds, resume_path=kwargs['model_ckpt_path'])
        except RuntimeError:
            model, _ = make_and_restore_model(arch=model, dataset=rb_ds, resume_path=kwargs['model_ckpt_path'])
  
  model.cuda(kwargs['gpu_id'])
  model.eval()

  if kwargs['objective_type'] == 'logit':
    custom_loss = logit_loss
  elif kwargs['objective_type'] == 'logit_diverge':
    present_classes = list(dataset.df[class_num_col].unique())
    df_for_class_to_class_num = pd.read_csv(kwargs['dirmap_path'])
    class_to_class_num = dict(zip(df_for_class_to_class_num['class'], df_for_class_to_class_num[class_num_col]))
    del df_for_class_to_class_num
    if kwargs['diverge_from'] and len(kwargs['diverge_from']) > 0:
      div_class_dict = {c: [class_to_class_num[c] for c in kwargs['diverge_from']] for c in present_classes}
    else: # By default, diverge from all other classes that are present
      div_class_dict = {c: [cd for cd in present_classes if c != cd] for c in present_classes}
    custom_loss = DivergentLogitLoss(div_class_dict)
  else:
    custom_loss = None

  perturb_kwargs = {
    'constraint': '2',
    'eps': kwargs['eps'],
    'step_size': kwargs['step_size'],
    'iterations': kwargs['num_steps'],
    'targeted': not kwargs['attack'],
    'do_tqdm': True,
    'custom_loss': custom_loss
  }

  print(perturb_kwargs)

  for i, data in enumerate(val_loader):

    if kwargs['dirmap_path']:
      im, label, idx = data
    else:
      im, label = data
      idx = None  # For predefined datasets, idx might not be used

    im = im.cuda(kwargs['gpu_id'])
    label = label.cuda(kwargs['gpu_id'])

    if kwargs['make_gifs'] and not kwargs['gif_extremes']:
      _, im_adv, all_im_tensors_for_gif = model(im, label, make_adv=True, get_all_step_ims=kwargs['make_gifs'], **perturb_kwargs)
    else:
      _, im_adv = model(im, label, make_adv=True, **perturb_kwargs)

    im_adv = im_adv.cpu()

    for j, img in enumerate(im_adv):
      
      if kwargs['dirmap_path']:
        im_path = dataset.df.loc[idx[j], "im_path"]
        img_class = dataset.df.loc[idx[j], "class"]
        if kwargs['diverge_from'] and len(kwargs['diverge_from']) > 0:
          fname_base, fname_ext = os.path.splitext(im_path)
          if kwargs['attack']:
            fname_base = fname_base + "_more_" + '_'.join(kwargs['diverge_from']) + "_less_" + img_class + "_eps_" + str(eps_fname)
          else:
            fname_base = fname_base + "_more_" + img_class + "_less_" + '_'.join(kwargs['diverge_from']) + "_eps_" + str(eps_fname)
          im_path = fname_base + fname_ext
          dataset.df.loc[idx[j], "im_path"] = im_path
        save_path = os.path.join(dest_dir, im_path)

        if not os.path.exists(os.path.dirname(save_path)):
          os.makedirs(os.path.dirname(save_path))

      else:
        save_path = os.path.join(dest_dir, f'image_{i}_{j}_adv.png')
      save_image(img, save_path)

      
      if kwargs['save_originals']:
        orig_img = im.cpu()[j, :, :, :]

        if kwargs['dirmap_path']:
          im_path = dataset.df.loc[idx[j], "im_path"]
          orig_save_path = os.path.join(orig_dest_dir, im_path)

          if not os.path.exists(os.path.dirname(orig_save_path)):
            os.makedirs(os.path.dirname(orig_save_path))
        else:
          orig_save_path = os.path.join(orig_dest_dir, f'image_{i}_{j}_aaa_orig.png')
        save_image(orig_img, orig_save_path)

      if kwargs['make_gifs']:
          
          if kwargs['gif_extremes']:
            all_im_tensors_for_gif = [im.detach().cpu(), im_adv.detach().cpu()]

          gif_images = []
          for batch_tensor in all_im_tensors_for_gif:
              im_tensor = batch_tensor[j, :, :, :]
              # Normalize to [0, 1]
              im_tensor = (im_tensor - im_tensor.min()) / (im_tensor.max() - im_tensor.min())
              # Convert to PIL Image to handle rescaling to [0, 255] and channel order
              image = TF.to_pil_image(im_tensor)
              # Convert PIL Image to numpy array
              numpy_image = np.array(image)
              # The numpy array is now in the correct format (HxWxC) and dtype (uint8)
              gif_images.append(numpy_image)

          if kwargs['gif_border']:
            num_frames = len(gif_images)
            border_width = 2  # Border width in pixels

            for i, img in enumerate(gif_images):
                color = compute_border_color(i, num_frames)
                gif_images[i] = add_border(gif_images[i], color, border_width=border_width)

          if kwargs['gif_forward_backward_loop']:
            reversed_gif_images = gif_images[1:-1][::-1] # Exclude first and last frames
            gif_images.extend(reversed_gif_images)

          base_save_path, _ = os.path.splitext(save_path)
          gif_save_path = base_save_path + ".gif"
          imageio.mimsave(gif_save_path, gif_images, fps=kwargs['gif_fps'], loop=0)

          # Generate and plot a histogram of these mean absolute pixel value differences
          # (abs. differences are averaged across channels within each pixel location)
          # mean_absolute_differences = np.mean(np.abs(gif_images[0] - gif_images[-1]), axis=2)
          mean_absolute_differences = np.abs(gif_images[0] - gif_images[-1])
          plt.figure()
          plt.hist(mean_absolute_differences.ravel(), bins=256, range=(0, 255), color='blue', alpha=0.7)
          plt.xlabel('Mean Absolute Difference')
          plt.ylabel('Frequency')
          base_save_path, _ = os.path.splitext(save_path)
          plt.savefig(base_save_path + "_hist.jpg", format='jpg')
          plt.close()

          plt.figure()
          plt.hist(gif_images[0].ravel(), bins=256, range=(0, 255), color="red", alpha=0.7)
          plt.xlabel('Pixel value')
          plt.ylabel('Frequency')
          plt.savefig(base_save_path + "_hist_orig.jpg", format='jpg')
          plt.close()

          plt.figure()
          plt.hist(gif_images[-1].ravel(), bins=256, range=(0, 255), color="green", alpha=0.7)
          plt.xlabel('Pixel value')
          plt.ylabel('Frequency')
          plt.savefig(base_save_path + "_hist_adv.jpg", format='jpg')
          plt.close()


    if kwargs['dirmap_path']:
      # Save the updated DataFrame
      new_csv_path = os.path.join(dest_dir, "dirmap.csv")
      dataset.df.to_csv(new_csv_path, index=False)

      if kwargs['save_originals']:
        # It is the same dataframe for the original natural images - just in a different directory. 
        new_csv_path = os.path.join(orig_dest_dir, "dirmap.csv")
        dataset.df.to_csv(new_csv_path, index=False)


if __name__ == "__main__":
  args = get_args(sys.argv[1:])

  start = time.time()
  enhance_images(**args)
  print("Runtime in seconds:", time.time() - start)