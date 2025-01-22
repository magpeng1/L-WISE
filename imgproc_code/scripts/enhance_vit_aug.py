
"""Enhance an entire dataset of images using a neural model, with multi-view augmentations.
Run from root project folder using "python scripts/enhance_vit_aug.py ..."
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent)) # Add parent directory to pythonpath
import argparse
from tqdm import trange
import random
import pandas as pd
from matplotlib import pyplot as plt
import torch as ch
import time
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from robustness.datasets import *
from robustness.model_utils import make_and_restore_model
from utils.dirmap_dataset import DirmapDataset, custom_collate
from utils.mapped_models import imagenet_mapped_model
from utils.custom_losses import *
from utils.gif_utils import *
from robustness.tools.constants import IMAGENET_16_RANGES as i16_ranges
import robustness
from robustness import attack_steps  # Assuming attack_steps.py is in the PYTHONPATH
from robustness.attacker import AttackerModel  # Assuming attacker.py is in the PYTHONPATH
from robustness.imagenet_models import *


def get_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dest_dir', type=str, required=True, help="Location at which dest dir will be created")
    parser.add_argument('--dirmap_path', type=str, default=None, help="Path to dirmap csv.")
    parser.add_argument('--dataset_path', type=str, default="/media/KLAB37/datasets/ImageNet2012", help="Path to dataset")
    parser.add_argument('--dataset_name', type=str, default=None, required=False, help="Name of dataset as defined in Robustness library (see robustness/datasets.py). Or, one of the superclass subset names (i.e. imagenet16). Can leave unspecified if using superclasses.")
    parser.add_argument('--model_ckpt_path', type=str, default="/home/morgan/projects/learn-histopath-backend/model_ckpts/imagenet_l2_3_0.pt", help="Path to model checkpoint")
    parser.add_argument('--arch', type=str, default='resnet50', help="Name of CNN archhitecture (e.g. resnet50, convnext_tiny, densenet201)")
    parser.add_argument('--from_tf_harmonized', default=False, action='store_true', help="Add this flag if you are using harmonized ViT **does not currently work properly")

    # Setting up the optimization
    parser.add_argument('--eps', default=10, type=float, help="Epsilon budget for perturbation")
    parser.add_argument('--step_size', default=0.5, type=float, help="Perturbation step size")
    parser.add_argument('--num_steps', default=20, type=int, help="Number of steps for perturbation")
    parser.add_argument('--objective_type', type=str, default="logit", help="Type of objective for enhancement. cross_entropy | logit | logit_diverge")
    parser.add_argument('--diverge_from', nargs='+', help="Specific classes to diverge from, if using --objective_type logit_diverge. E.g., --diverge_from class1 class2 class3. All non-groundtruth classes are diverged from by default")
    parser.add_argument('--superclass', type=str, default=None, help="Optimize toward superclass label instead of fine-grained class label. restrictedimagenet | imagenet16")
    parser.add_argument('--attack', default=False, action='store_true', help="Do an attack instead of enhancement")
    parser.add_argument('--num_augs', default=4, type=int, help="Number of augmentations for perturbation")
    parser.add_argument('--aug_types', nargs="+", default=["color", "translation", "resize", "cutout"], help="List the type of augmentations, which can be color, translation, resize, cutout. E.g. --aug_types color resize")

    parser.add_argument('--save_originals', default=False, action='store_true', help="Save original images")

    parser.add_argument('--num_workers', type=int, default=1, help="Number of CPU threads for dataloader")
    parser.add_argument('--gpu_id', type=int, default=0, help="ID of GPU to use")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")

    parser.add_argument('--make_gifs', default=False, action='store_true', help="Make a GIF of each enhancement")
    parser.add_argument('--gif_forward_backward_loop', default=False, action='store_true', help="GIFs move in backwards and forwards directions.")
    parser.add_argument('--gif_border', default=False, action='store_true', help="Make a color-changing border for the gif.")
    parser.add_argument('--gif_extremes', default=False, action='store_true', help="Store only the extremes of the image. If using this flag, best to set --gif_fps 1")
    parser.add_argument('--gif_fps', type=int, default=15, help="Frames per second for gif generation")

    if argv is None:
        args = argparse.Namespace()

        for action in parser._actions:
            if action.default is not argparse.SUPPRESS:
                setattr(args, action.dest, action.default)

        args_dict = vars(args)
    else:
        args = parser.parse_args(argv)
        args_dict = vars(args)

    return args_dict

# Data Augmentation Functions
def rand_brightness(x):
    factor = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 0.4 + 0.8  # Brightness factor between 0.8 and 1.2
    return x * factor

def rand_saturation(x):
    mean = x.mean(dim=1, keepdim=True)
    factor = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 0.4 + 0.8  # Saturation factor between 0.8 and 1.2
    return (x - mean) * factor + mean

def rand_contrast(x):
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    factor = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 0.4 + 0.8  # Contrast factor between 0.8 and 1.2
    return (x - mean) * factor + mean

def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()
    return x

def rand_resize(x, min_ratio=0.8, max_ratio=1.2):
    resize_ratio = torch.rand(1).item() * (max_ratio - min_ratio) + min_ratio
    resized_img = F.interpolate(x, size=int(resize_ratio * x.shape[3]), mode='bilinear', align_corners=False)
    org_size = x.shape[3]
    if resized_img.shape[2] < org_size:
        padding = (org_size - resized_img.shape[2]) // 2
        x = F.pad(resized_img, (padding, org_size - resized_img.shape[2] - padding, padding, org_size - resized_img.shape[3] - padding), "constant", 0.)
    else:
        x = resized_img[:, :, :org_size, :org_size]
    return x

def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    mask = torch.ones_like(x)
    for img in range(x.size(0)):
        top = random.randint(0, x.size(2) - cutout_size[0]) if x.size(2) > cutout_size[0] else 0
        left = random.randint(0, x.size(3) - cutout_size[1]) if x.size(3) > cutout_size[1] else 0
        mask[img, :, top:top + cutout_size[0], left:left + cutout_size[1]] = 0.
    return x * mask

AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'resize': [rand_resize],
    'cutout': [rand_cutout],
}

def DiffAugment(x, policy='color,translation,resize,cutout', channels_first=True):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x

def modify_gradients_spatial(grad, kernel_size=7, contrast_factor=2.0, eps=1e-6):
    """
    Modifies gradients based on local spatial characteristics.
    Areas with high local gradient magnitudes are enhanced more than areas with low local magnitudes.
    
    Args:
        grad (torch.Tensor): Input gradient tensor of shape [batch, channels, height, width]
        kernel_size (int): Size of the neighborhood to consider (must be odd)
        contrast_factor (float): Controls strength of the enhancement
        eps (float): Small value for numerical stability
    
    Returns:
        torch.Tensor: Modified gradient tensor
    """
    # Compute gradient magnitudes
    magnitudes = torch.abs(grad)
    
    # Create a Gaussian kernel for weighted averaging
    sigma = kernel_size / 6.0  # This makes the kernel effectively zero at edges
    x = torch.arange(-(kernel_size//2), kernel_size//2 + 1, device=grad.device)
    gaussian_1d = torch.exp(-x**2 / (2 * sigma**2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    
    # Create 2D Gaussian kernel
    kernel_2d = gaussian_1d.unsqueeze(0) * gaussian_1d.unsqueeze(1)
    kernel = kernel_2d.expand(grad.size(1), 1, kernel_size, kernel_size)
    
    # Compute local average magnitudes using convolution
    # We need to process each batch item separately to avoid edge effects
    local_avg_magnitudes = torch.zeros_like(magnitudes)
    for b in range(grad.size(0)):
        local_avg_magnitudes[b] = F.conv2d(
            magnitudes[b:b+1],
            kernel.to(grad.device),
            padding=kernel_size//2,
            groups=grad.size(1)
        )
    
    # Normalize local averages to [0, 1] per batch item
    local_avg_normalized = local_avg_magnitudes / (local_avg_magnitudes.amax(dim=(1,2,3), keepdim=True) + eps)
    
    # Create enhancement factors based on local magnitude averages
    enhancement_factors = 1.0 + (contrast_factor - 1.0) * local_avg_normalized
    
    # Apply enhancement while preserving signs
    signs = torch.sign(grad)
    enhanced_magnitudes = magnitudes * enhancement_factors
    
    return signs * enhanced_magnitudes

def enhance_images(**kwargs):
    # Get defaults if this is being called as a function by an outside script
    if not __name__ == "__main__":
        default_kwargs = get_args(None)
        default_kwargs.update(kwargs)
        kwargs = default_kwargs

    eps_fname = int(kwargs['eps']) if float(kwargs['eps']).is_integer() else kwargs['eps']
    step_size_fname = int(kwargs['step_size']) if float(kwargs['step_size']).is_integer() else kwargs['step_size']

    perturb_id = f"{eps_fname}_{step_size_fname}_{kwargs['num_steps']}_{kwargs['objective_type']}"

    if kwargs['diverge_from'] and len(kwargs['diverge_from']) > 0:
        perturb_id += "_from_" + '_'.join(kwargs['diverge_from'])

    if kwargs['superclass']:
        perturb_id += f"_{kwargs['superclass']}"
    if kwargs['attack']:
        perturb_id += "_attack"

    if kwargs['dirmap_path']:
        dest_dir_name = os.path.basename(os.path.dirname(kwargs['dirmap_path'])) + "_" + perturb_id
        dirmap = pd.read_csv(kwargs['dirmap_path'])
        if kwargs['diverge_from'] and len(kwargs['diverge_from']) > 0:
            dirmap = dirmap[~dirmap['class'].isin(kwargs['diverge_from'])].reset_index(drop=True)
    else:
        dest_dir_name = perturb_id

    dest_dir = os.path.join(kwargs['dest_dir'], dest_dir_name)

    print("Saving to:", dest_dir)

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    if kwargs['save_originals']:
        orig_dest_dir_name = os.path.basename(os.path.dirname(kwargs['dirmap_path'])) + "_natural" if kwargs['dirmap_path'] else "natural"
        orig_dest_dir = os.path.join(kwargs['dest_dir'], orig_dest_dir_name)
        
        if not os.path.exists(orig_dest_dir):
            os.makedirs(orig_dest_dir)

    if kwargs['superclass']:
        if kwargs['superclass'].lower() in ["imagenet16"]:
            rb_ds = CustomImageNet(kwargs['dataset_path'], list(i16_ranges.values()))
        else:
            raise ValueError(f"Undefined superclass setting \"{kwargs['superclass']}\"")
    else:
        rb_ds = eval(f"{kwargs['dataset_name']}(\"{kwargs['dataset_path']}\")")

    if kwargs['dirmap_path']:
        transform = Compose([
            transforms.Resize(224),
            transforms.CenterCrop((224, 224)),
            ToTensor()
        ])
        class_num_col = "orig_class_num" if ('ImageNet' in kwargs['dataset_name']) else "class_num"
        dataset = DirmapDataset(csv_file=dirmap, transform=transform, class_num_col=class_num_col, dataset_path=kwargs['dataset_path'], mean=ch.tensor([0.485, 0.456, 0.406]), std=ch.tensor([0.229, 0.224, 0.225]))
        val_loader = DataLoader(dataset, batch_size=kwargs['batch_size'], num_workers=kwargs['num_workers'], collate_fn=custom_collate)
    else:
        _, val_loader = rb_ds.make_loaders(workers=kwargs['num_workers'], batch_size=kwargs['batch_size'])

    # Initialize model
    if 'xcit' in kwargs['arch']:
        xcit_model = eval(kwargs['arch'] + "(pretrained=False)")
        state_dict = ch.load(kwargs['model_ckpt_path'])
        xcit_model.load_state_dict(state_dict)
        model, _ = make_and_restore_model(arch=xcit_model, dataset=rb_ds, gpu_id=kwargs['gpu_id'])
    else:
        if kwargs['arch'] == 'vit':
            num_classes = 1000 if ('ImageNet' in kwargs['dataset_name']) else rb_ds.num_classes
            arch = robustness.imagenet_models.vit.create_vit_model(num_classes=num_classes, pretrained=False, from_tf=kwargs['from_tf_harmonized'])
        else:
            arch = kwargs['arch']

        if kwargs['superclass']:
            model, _ = imagenet_mapped_model(arch=arch, superclassed_imagenet_ds=rb_ds, pytorch_pretrained=False, gpu_id=kwargs['gpu_id'], resume_path=kwargs['model_ckpt_path'])
        else:
            try:
                model, _ = make_and_restore_model(arch=arch, dataset=rb_ds, gpu_id=kwargs['gpu_id'], resume_path=kwargs['model_ckpt_path'])
            except RuntimeError:
                model, _ = make_and_restore_model(arch=arch, dataset=rb_ds, gpu_id=kwargs['gpu_id'])
                try:
                    model, _ = make_and_restore_model(arch=model.model, dataset=rb_ds, resume_path=kwargs['model_ckpt_path'])
                except RuntimeError:
                    model, _ = make_and_restore_model(arch=model, dataset=rb_ds, resume_path=kwargs['model_ckpt_path'])
    
    model.cuda(kwargs['gpu_id'])
    model.eval()

    # Select loss function
    if kwargs['objective_type'] == 'logit':
        custom_loss = logit_loss
    elif kwargs['objective_type'] == 'logit_diverge':
        present_classes = list(dataset.df[class_num_col].unique()) if kwargs['dirmap_path'] else list(range(rb_ds.num_classes))
        df_for_class_to_class_num = pd.read_csv(kwargs['dirmap_path']) if kwargs['dirmap_path'] else None
        class_to_class_num = dict(zip(df_for_class_to_class_num['class'], df_for_class_to_class_num[class_num_col])) if df_for_class_to_class_num is not None else {c: c for c in present_classes}
        del df_for_class_to_class_num
        if kwargs['diverge_from'] and len(kwargs['diverge_from']) > 0:
            div_class_dict = {c: [class_to_class_num[c] for c in kwargs['diverge_from']] for c in present_classes}
        else:
            div_class_dict = {c: [cd for cd in present_classes if c != cd] for c in present_classes}
        custom_loss = DivergentLogitLoss(div_class_dict)
    else:
        custom_loss = None

    # Initialize AttackerModel
    attacker_model = AttackerModel(model, rb_ds if not kwargs['dirmap_path'] else dataset)
    attacker_model.cuda(kwargs['gpu_id'])

    # Set up perturbation parameters
    perturb_kwargs = {
        'constraint': '2',
        'eps': kwargs['eps'],
        'step_size': kwargs['step_size'],
        'iterations': kwargs['num_steps'],
        'targeted': not kwargs['attack'],
        'do_tqdm': True,
        'custom_loss': custom_loss
    }

    aug_policy = ",".join(kwargs['aug_types'])

    print(perturb_kwargs)

    # Initialize step class
    STEPS = {
        'inf': attack_steps.LinfStep,
        '2': attack_steps.L2Step,
        'unconstrained': attack_steps.UnconstrainedStep,
        'fourier': attack_steps.FourierStep,
        'random_smooth': attack_steps.RandomStep
    }

    m = -1 if perturb_kwargs['targeted'] else 1

    for i, data in enumerate(val_loader):
        if kwargs['dirmap_path']:
            im, label, idx = data
        else:
            im, label = data
            idx = None

        im = im.cuda(kwargs['gpu_id'])
        label = label.cuda(kwargs['gpu_id'])

        step_class = STEPS[perturb_kwargs['constraint']]
        step = step_class(eps=perturb_kwargs['eps'], orig_input=im.detach().cuda(), model=model, step_size=perturb_kwargs['step_size'])

        # Random start
        if perturb_kwargs.get('random_start', False):
            im = step.random_perturb(im)

        im_adv = im

        for _ in trange(perturb_kwargs['iterations']):
            im_adv = im_adv.clone().detach().requires_grad_(True)
            
            # Forward pass with multiple augmentations
            losses = []
            for _ in range(kwargs['num_augs']):  # num_augmentations
                im_aug = DiffAugment(im_adv, policy=aug_policy, channels_first=True)
                try:
                    loss, _ = custom_loss(model.model, step.to_image(im_aug), label) if custom_loss else F.cross_entropy(model.model(im_aug), label)
                except AttributeError:
                    loss, _ = custom_loss(model, step.to_image(im_aug), label) if custom_loss else F.cross_entropy(model(im_aug), label)
                losses.append(ch.mean(loss))

            loss = torch.stack(losses).mean()

            # Backward pass
            #grad, = torch.autograd.grad(m * loss, [im_adv])
            grad = torch.autograd.grad(m * loss, im_adv, create_graph=False)[0]

            #grad = modify_gradients_spatial(grad)

            with torch.no_grad():
                im_adv = step.step(im_adv, grad)
                im_adv = step.project(im_adv)

        # Save the perturbed images
        for j, img in enumerate(im_adv):
            if kwargs['dirmap_path']:
                im_path = dataset.df.loc[idx[j], "im_path"]
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
                    orig_save_path = os.path.join(orig_dest_dir, f'image_{i}_{j}_orig.png')
                save_image(orig_img, orig_save_path)

            if kwargs['make_gifs']:
                gif_images = [im[j].cpu(), im_adv[j]]
                if kwargs['gif_border']:
                    num_frames = len(gif_images)
                    border_width = 2
                    gif_images = [add_border(img, compute_border_color(k, num_frames), border_width=border_width) for k, img in enumerate(gif_images)]
                if kwargs['gif_forward_backward_loop']:
                    reversed_gif_images = gif_images[1:-1][::-1]
                    gif_images.extend(reversed_gif_images)
                base_save_path, _ = os.path.splitext(save_path)
                gif_save_path = base_save_path + ".gif"
                gif_pil_images = [TF.to_pil_image(img) for img in gif_images]
                gif_pil_images[0].save(gif_save_path, save_all=True, append_images=gif_pil_images[1:], fps=kwargs['gif_fps'], loop=0)

                # Generate and plot histograms
                mean_absolute_differences = (im_adv[j] - im[j].cpu()).abs().numpy()
                plt.figure()
                plt.hist(mean_absolute_differences.ravel(), bins=256, range=(0, 1), color='blue', alpha=0.7)
                plt.xlabel('Mean Absolute Difference')
                plt.ylabel('Frequency')
                plt.savefig(base_save_path + "_hist.jpg", format='jpg')
                plt.close()

                plt.figure()
                plt.hist(im[j].cpu().numpy().ravel(), bins=256, range=(0, 1), color="red", alpha=0.7)
                plt.xlabel('Pixel value')
                plt.ylabel('Frequency')
                plt.savefig(base_save_path + "_hist_orig.jpg", format='jpg')
                plt.close()

                plt.figure()
                plt.hist(im_adv[j].numpy().ravel(), bins=256, range=(0, 1), color="green", alpha=0.7)
                plt.xlabel('Pixel value')
                plt.ylabel('Frequency')
                plt.savefig(base_save_path + "_hist_adv.jpg", format='jpg')
                plt.close()

        if kwargs['dirmap_path']:
            new_csv_path = os.path.join(dest_dir, "dirmap.csv")
            dataset.df.to_csv(new_csv_path, index=False)

            if kwargs['save_originals']:
                new_csv_path = os.path.join(orig_dest_dir, "dirmap.csv")
                dataset.df.to_csv(new_csv_path, index=False)

if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    start = time.time()
    enhance_images(**args)
    print("Runtime in seconds:", time.time() - start)