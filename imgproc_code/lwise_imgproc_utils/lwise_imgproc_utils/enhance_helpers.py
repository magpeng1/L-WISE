import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import os

# Blue->red heatmap with alpha blending
def create_heatmap(diff_tensor, smooth_kernel_size=None):
    """
    Creates a heatmap from difference tensor.
    Args:
        diff_tensor: Tensor of shape [C, H, W]
        smooth_kernel_size: Optional int for gaussian smoothing kernel size
    Returns:
        Tensor of shape [3, H, W] representing RGB heatmap
    """
    # Calculate magnitude of difference across channels
    if diff_tensor.shape[0] == 3:  # RGB image
        magnitude = torch.sqrt(torch.sum(diff_tensor ** 2, dim=0))
    else:
        magnitude = torch.abs(diff_tensor.squeeze())
    
    # Optional smoothing
    if smooth_kernel_size:
        kernel_size = smooth_kernel_size
        sigma = kernel_size / 6.0
        channels = 1
        kernel = torch.ones(channels, 1, kernel_size, kernel_size) / (kernel_size * kernel_size)
        magnitude = magnitude.unsqueeze(0).unsqueeze(0)
        magnitude = F.conv2d(magnitude, kernel.to(magnitude.device), padding=kernel_size//2)
        magnitude = magnitude.squeeze()
    
    # Normalize to [0, 1]
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
    
    # Create RGB heatmap (red is high difference, blue is low)
    heatmap = torch.zeros(3, magnitude.shape[0], magnitude.shape[1])
    heatmap[0] = magnitude  # Red channel
    heatmap[2] = 1 - magnitude  # Blue channel
    
    return heatmap

def create_overlay(base_image, heatmap, alpha=0.7):
    """
    Overlays heatmap on base image with transparency.
    Args:
        base_image: Tensor of shape [C, H, W]
        heatmap: Tensor of shape [3, H, W]
        alpha: Float transparency factor (0 to 1)
    Returns:
        Tensor of shape [3, H, W] with overlay
    """
    # Ensure base image is in range [0, 1]
    base_image = (base_image - base_image.min()) / (base_image.max() - base_image.min())
    
    # Combine using alpha blending
    overlay = (1 - alpha) * base_image + alpha * heatmap
    
    # Ensure output is in valid range
    overlay = torch.clamp(overlay, 0, 1)
    
    return overlay


def save_visualization_suite(im, im_adv, save_path, smooth_kernel_size=None):
    """
    Saves full suite of visualizations: difference image, heatmap, and overlaid versions.
    Args:
        im: Original image tensor [C, H, W]
        im_adv: Enhanced image tensor [C, H, W]
        save_path: Base path for saving images
        smooth_kernel_size: Optional kernel size for smoothing heatmap
    """
    base_save_path, ext = os.path.splitext(save_path)
    
    # Calculate difference
    im_diff = im_adv.detach().cpu() - im.detach().cpu()
    
    # Save normalized difference image
    im_diff_norm = (im_diff - im_diff.min()) / (im_diff.max() - im_diff.min())
    save_image(im_diff_norm, base_save_path + "_DIFF" + ext)
    
    # Create and save heatmap
    heatmap = create_heatmap(im_diff, smooth_kernel_size)
    save_image(heatmap, base_save_path + "_HEATMAP" + ext)
    
    # Create and save overlaid versions
    overlay_orig = create_overlay(im, heatmap)
    save_image(overlay_orig, base_save_path + "_HEATMAP_OVERLAID_ORIG" + ext)
    
    overlay_enh = create_overlay(im_adv, heatmap)
    save_image(overlay_enh, base_save_path + "_HEATMAP_OVERLAID_ENH" + ext)