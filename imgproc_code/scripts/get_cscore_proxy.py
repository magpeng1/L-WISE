import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent)) # Add parent directory to pythonpath
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from robustness.model_utils import make_and_restore_model
from robustness.datasets import *
from tqdm import tqdm
import json
from utils.dirmap_dataset import DirmapDataset, custom_collate

def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='ImageNet', help='Name of dataset as defined in Robustness library')
    parser.add_argument('--dirmap_path', type=str, required=True, help="Path to dirmap CSV file")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to dataset")
    parser.add_argument('--model_dir', type=str, required=True, help="Directory containing model checkpoints")
    parser.add_argument('--checkpoint_filename_pattern', type=str, default="#_checkpoint.pt", help="Pattern for checkpoint filenames")
    parser.add_argument('--min_epoch', type=int, default=0, help="Minimum epoch to consider")
    parser.add_argument('--max_epoch', type=int, default=None, help="Maximum epoch to consider")
    parser.add_argument('--epoch_frequency', type=int, default=1, help="Frequency of epochs to use")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for evaluation")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for data loading")
    parser.add_argument('--gpu_id', type=int, default=0, help="ID of GPU to use")
    parser.add_argument('--json_output_path', type=str, required=True, help="Path for JSON output file")
    
    args = parser.parse_args(argv)
    return vars(args)

def get_checkpoint_epochs(model_dir, pattern, min_epoch, max_epoch):
    epochs = []
    for filename in os.listdir(model_dir):
        if pattern.replace('#', '') in filename:
            try:
                epoch = int(filename.split('_')[0])
                if (min_epoch is None or epoch >= min_epoch) and (max_epoch is None or epoch <= max_epoch):
                    epochs.append(epoch)
            except ValueError:
                continue
    return sorted(epochs)

def calculate_c_score_proxy(args):
    # Set up dataset and data loader
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor()
    ])
    if args['dataset_name'] in ["ImageNet"]:
      class_num_col = "orig_class_num"
    else:
      class_num_col = "class_num"
    dataset = DirmapDataset(csv_file=args['dirmap_path'], transform=transform, dataset_path=args['dataset_path'], class_num_col=class_num_col)
    rb_ds = eval(f"{args['dataset_name']}(\"{args['dataset_path']}\")")
    val_loader = DataLoader(dataset, batch_size=args['batch_size'], num_workers=args['num_workers'], collate_fn=custom_collate)

    # Get checkpoint epochs
    epochs = get_checkpoint_epochs(args['model_dir'], args['checkpoint_filename_pattern'], 
                                   args['min_epoch'], args['max_epoch'])
    
    # Apply epoch frequency
    epochs = epochs[::args['epoch_frequency']]
    
    print(f"Using epochs: {epochs}")
    print(f"Number of checkpoints to be used: {len(epochs)}")
    
    if len(epochs) % args['epoch_frequency'] != 0:
        print("Warning: epoch_frequency is not a factor of the number of epochs. Some checkpoints at the end won't be used.")

    # Initialize c-score proxy dictionary
    c_scores = {i: 0 for i in range(len(dataset))}

    # Iterate through checkpoints
    for epoch in tqdm(epochs, desc="Processing epochs"):
        checkpoint_path = os.path.join(args['model_dir'], args['checkpoint_filename_pattern'].replace('#', str(epoch)))
        
        # Load model
        model, _ = make_and_restore_model(arch='resnet50', dataset=rb_ds, resume_path=checkpoint_path)
        model = model.cuda(args['gpu_id'])
        model.eval()

        # Evaluate images
        correct_preds = set()
        for images, labels, idxs in tqdm(val_loader, desc=f"Evaluating epoch {epoch}", leave=False):
            images = images.cuda(args['gpu_id'])
            labels = labels.cuda(args['gpu_id'])
            
            with torch.no_grad():
                outputs, _ = model(images)
                _, predicted = outputs.max(1)
                
                correct = predicted.eq(labels).cpu().numpy()
                
                # Use boolean indexing to get correct idxs
                correct_idxs = [idx for idx, is_correct in zip(idxs, correct) if is_correct]
                correct_preds.update(correct_idxs)

        # Update c-scores
        for idx in correct_preds:
            c_scores[idx] += 1

    # Normalize c-scores
    num_epochs = len(epochs)
    for idx in c_scores:
        c_scores[idx] /= num_epochs

    return c_scores, dataset

if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    c_scores, dataset = calculate_c_score_proxy(args)
    
    # Prepare JSON output
    json_output = {}
    for idx, score in c_scores.items():
        image_path = dataset.df.iloc[idx]['im_path']
        json_output[image_path] = score

    # Save JSON output
    with open(args['json_output_path'], 'w') as f:
        json.dump(json_output, f, indent=2)

    print(f"C-score proxies calculated and saved to {args['json_output_path']}")
    print("First 10 scores:")
    for filename, score in list(json_output.items())[:10]:
        print(f"{filename}: {score}")