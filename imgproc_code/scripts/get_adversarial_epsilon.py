"""Calculate adversarial epsilon for images in a dataset using FGSM.
Run from root project folder using "python scripts/get_adversarial_epsilon.py ..."
"""

import sys
import argparse
import json
import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from robustness.datasets import *
from robustness.model_utils import make_and_restore_model
from lwise_imgproc_utils.dirmap_dataset import DirmapDataset, custom_collate

def get_args(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_name', type=str, required=True, help="Name of dataset as defined in Robustness library")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to dataset")
    parser.add_argument('--dirmap_path', type=str, default=None, help="Path to dirmap csv.")
    parser.add_argument('--model_ckpt_path', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--json_output_path', type=str, required=True, help="Path to save output JSON")
    
    parser.add_argument('--num_workers', type=int, default=4, help="Number of CPU threads for dataloader")
    parser.add_argument('--gpu_id', type=int, default=0, help="ID of GPU to use")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    
    parser.add_argument('--disrupt_min_eps', type=float, default=0, help="Minimum epsilon for disruption")
    parser.add_argument('--disrupt_max_eps', type=float, default=2.5, help="Maximum epsilon for disruption")
    parser.add_argument('--disrupt_step_size', type=float, default=1.25e-5, help="Step size for disruption up to 0.02")
    parser.add_argument('--disrupt_large_step_size', type=float, default=0.005, help="Step size for disruption after 0.02")
    parser.add_argument('--disrupt_large_step_threshold', type=float, default=0.02, help="Threshold for switching to large step size")
    parser.add_argument('--disrupt_num_steps', type=int, default=1, help="Number of gradient steps for disruption")
    
    parser.add_argument('--correct_min_eps', type=float, default=0, help="Minimum epsilon for correction")
    parser.add_argument('--correct_max_eps', type=float, default=0.05, help="Maximum epsilon for correction")
    parser.add_argument('--correct_step_size', type=float, default=1.25e-6, help="Step size for correction up to 0.001")
    parser.add_argument('--correct_large_step_size', type=float, default=1.25e-5, help="Step size for correction after 0.001")
    parser.add_argument('--correct_large_step_threshold', type=float, default=0.001, help="Threshold for switching to large step size")
    parser.add_argument('--correct_num_steps', type=int, default=2, help="Number of gradient steps for correction")

    args = parser.parse_args(argv)
    return vars(args)

def fgsm_attack(model, image, label, epsilon, disrupt=True, loss_fn=F.cross_entropy):
    """
    Implements FGSM attack with support for both disruption and correction
    
    Args:
        model: The model to attack
        image: Input image
        label: True label
        epsilon: Step size
        disrupt: If True, try to cause misclassification (maximize loss)
                If False, try to achieve correct classification (minimize loss)
        loss_fn: Loss function to use
    """
    image.requires_grad = True
    
    # Forward pass
    output, _ = model(image)
    loss = loss_fn(output, label)
    
    # Calculate gradients
    model.zero_grad()
    loss.backward()
    
    # Create modified example
    # For disruption: add gradient sign to maximize loss
    # For correction: subtract gradient sign to minimize loss
    sign_data_grad = image.grad.sign()
    if disrupt:
        disrupted_image = image + epsilon * sign_data_grad
    else:
        disrupted_image = image - epsilon * sign_data_grad
    
    # Clamp to maintain [0,1] range
    disrupted_image = torch.clamp(disrupted_image, 0, 1)
    
    return disrupted_image

def calculate_adversarial_epsilon(model, image, label, disrupt=True, **kwargs):
    """
    Calculate the minimum epsilon needed for misclassification (disrupt=True)
    or correction (disrupt=False)
    """
    if disrupt:
        # Perturbation epsilon range (0 to 0.02 by 1.25e-5, then 0.02 to 2.5 by 0.005)
        epsilon_range = torch.cat([
            torch.arange(kwargs['disrupt_min_eps'], 
                        kwargs['disrupt_large_step_threshold'], 
                        kwargs['disrupt_step_size']),
            torch.arange(kwargs['disrupt_large_step_threshold'], 
                        kwargs['disrupt_max_eps'], 
                        kwargs['disrupt_large_step_size'])
        ])
        num_steps = kwargs['disrupt_num_steps']  # Should be 1
    else:
        # Correction epsilon range (0 to 0.001 by 1.25e-6, then 0.001 to 0.05 by 1.25e-5)
        epsilon_range = torch.cat([
            torch.arange(kwargs['correct_min_eps'], 
                        kwargs['correct_large_step_threshold'], 
                        kwargs['correct_step_size']),
            torch.arange(kwargs['correct_large_step_threshold'], 
                        kwargs['correct_max_eps'], 
                        kwargs['correct_large_step_size'])
        ])
        num_steps = kwargs['correct_num_steps']  # Should be 2

    # Move epsilon range to same device as model
    epsilon_range = epsilon_range.to(image.device)
    
    for epsilon in epsilon_range:
        disrupted_image = image.clone()
        
        # Apply gradient steps
        for _ in range(num_steps):
            disrupted_image = fgsm_attack(model, disrupted_image, label, epsilon, disrupt)
            disrupted_image = disrupted_image.detach()
        
        # Check if modification succeeded
        with torch.no_grad():
            output, _ = model(disrupted_image)
            pred = output.max(1, keepdim=True)[1]
            
            if disrupt and pred.item() != label.item():
                return epsilon.item()
            elif not disrupt and pred.item() == label.item():
                return -epsilon.item() # Return negative value to reflect correction. Intuitively, harder images have low eps
    
    return None

def main(**kwargs):
    device = torch.device(f"cuda:{kwargs['gpu_id']}" if torch.cuda.is_available() else "cpu")

    # Load dataset
    rb_ds = eval(f"{kwargs['dataset_name']}(\"{kwargs['dataset_path']}\")")
    if kwargs['dirmap_path']:
        transform = Compose([
            Resize((224, 224)),
            ToTensor()
        ])
        
        class_num_col = 'orig_class_num' if kwargs['dataset_name'] == "ImageNet" else 'class_num'
        dataset = DirmapDataset(
            csv_file=kwargs['dirmap_path'],
            transform=transform,
            dataset_path=kwargs['dataset_path'],
            class_num_col=class_num_col
        )
        data_loader = DataLoader(
            dataset,
            batch_size=kwargs['batch_size'],
            num_workers=kwargs['num_workers'],
            collate_fn=custom_collate,
            shuffle=False
        )
    else:
        _, data_loader = rb_ds.make_loaders(
            workers=kwargs['num_workers'],
            batch_size=kwargs['batch_size']
        )

    # Load model
    try:
        model, _ = make_and_restore_model(
            arch='resnet50',
            dataset=rb_ds,
            resume_path=kwargs['model_ckpt_path']
        )
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    results = {}
    
    try:
        for images, labels, idxs in tqdm.tqdm(data_loader):
            images, labels = images.to(device), labels.to(device)
            
            with torch.no_grad():
                outputs, _ = model(images)
                predictions = outputs.max(1)[1]
            
            for i, (image, label, pred, idx) in enumerate(zip(images, labels, predictions, idxs)):
                disrupt = (pred == label)
                epsilon = calculate_adversarial_epsilon(
                    model, 
                    image.unsqueeze(0), 
                    label.unsqueeze(0), 
                    disrupt, 
                    **kwargs
                )
                
                image_name = dataset.df.loc[idx, 'im_path'] if kwargs['dirmap_path'] else f'image_{idx}'
                results[image_name] = epsilon

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"GPU out of memory error: {e}")
            torch.cuda.empty_cache()
        else:
            print(f"Runtime error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Save results even if interrupted
        with open(kwargs['json_output_path'], 'w') as f:
            json.dump(results, f)

if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    main(**args)