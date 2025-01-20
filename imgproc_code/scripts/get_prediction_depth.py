import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent)) # Add parent directory to pythonpath
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop
from robustness.datasets import ImageNet
from robustness.model_utils import make_and_restore_model
from utils.dirmap_dataset import DirmapDataset, custom_collate
from tqdm import tqdm

def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='ImageNet', help='Name of dataset as defined in Robustness library')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--dirmap_path', type=str, default=None, help='Path to dirmap csv')
    parser.add_argument('--model_ckpt_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--train_probes', action='store_true', help='Train linear probes')
    parser.add_argument('--probe_save_path', type=str, default=None, help='Path to save/load linear probes')
    parser.add_argument('--json_output_path', type=str, required=True, help='Path to save JSON output')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    return parser.parse_args(argv)

class LinearProbe(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

def get_layer_features(model, x, layer_index):
    layers = list(model.model.children())
    layer_outputs = []
    
    # Initial layers
    x = layers[0](x)  # conv1
    x = layers[1](x)  # bn1
    x = layers[2](x)  # relu
    layer_outputs.append(x)  # Layer 0
    
    x = layers[3](x)  # maxpool
    layer_outputs.append(x)  # Layer 1
    
    # Residual blocks
    for i, layer in enumerate([layers[4], layers[5], layers[6], layers[7]]):
        for j, bottleneck in enumerate(layer):
            x = bottleneck(x)
            layer_outputs.append(x)
    
    if layer_index < len(layer_outputs):
        return layer_outputs[layer_index]
    else:
        return x

def train_probes(model, train_loader, val_loader, num_classes, device, probe_save_path, subset_fraction=1.0):
    probes = []
    criterion = nn.CrossEntropyLoss()

    # First, let's check the dimensions of features at each layer
    sample_batch, _ = next(iter(train_loader))
    sample_batch = sample_batch.to(device)
    for layer in range(16):
        with torch.no_grad():
            features = get_layer_features(model, sample_batch, layer)
        print(f"Layer {layer} feature shape: {features.shape}")

    for layer in range(16):
        with torch.no_grad():
            sample_features = get_layer_features(model, sample_batch, layer)
        in_features = sample_features.shape[1]  # Use the actual number of channels

        probe = LinearProbe(in_features, num_classes).to(device)
        optimizer = optim.Adam(probe.parameters(), lr=0.001)
        probes.append(probe)

        n_epochs = 3
        for epoch in range(n_epochs):  # Train for 3 epochs
            print(f"STARTING EPOCH {epoch+1}/{n_epochs} FOR LAYER {layer}")
            model.eval()
            probe.train()
            
            # Calculate how many batches to use
            total_batches = len(train_loader)
            batches_to_use = int(total_batches * subset_fraction)
            
            for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f'Layer {layer}, Epoch {epoch+1}/{n_epochs}', total=batches_to_use)):
                if batch_idx >= batches_to_use:
                    break  # Stop after processing the desired number of batches
                
                images, labels = images.to(device), labels.to(device)
                with torch.no_grad():
                    features = get_layer_features(model, images, layer)
                outputs = probe(features)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Validate
        model.eval()
        probe.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                features = get_layer_features(model, images, layer)
                outputs = probe(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy of probe at layer {layer}: {100 * correct / total}%')

    # Save probes
    os.makedirs(probe_save_path, exist_ok=True)
    for i, probe in enumerate(probes):
        torch.save(probe.state_dict(), os.path.join(probe_save_path, f'probe_{i}.pth'))

def train_probes_parallel(model, train_loader, val_loader, num_classes, device, probe_save_path, num_layers=16, subset_fraction=1.0):
    probes = []
    optimizers = []
    criterion = nn.CrossEntropyLoss()

    # Create probes and optimizers for all layers
    for layer in range(num_layers):
        sample_batch, _ = next(iter(train_loader))
        sample_batch = sample_batch.to(device)
        with torch.no_grad():
            sample_features = get_layer_features(model, sample_batch, layer)
        in_features = sample_features.shape[1]
        probe = LinearProbe(in_features, num_classes).to(device)
        optimizer = optim.Adam(probe.parameters(), lr=0.001)
        probes.append(probe)
        optimizers.append(optimizer)

    n_epochs = 10
    total_batches = len(train_loader)
    batches_to_use = int(total_batches * subset_fraction)

    for epoch in range(n_epochs):
        print(f"STARTING EPOCH {epoch+1}/{n_epochs}")
        model.eval()
        for probe in probes:
            probe.train()

        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}', total=batches_to_use)):
            if batch_idx >= batches_to_use:
                break

            images, labels = images.to(device), labels.to(device)

            # Get features for all layers
            with torch.no_grad():
                features = [get_layer_features(model, images, layer) for layer in range(num_layers)]

            # Train all probes
            for layer, (probe, optimizer) in enumerate(zip(probes, optimizers)):
                outputs = probe(features[layer])
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Validate all probes
        for layer, probe in enumerate(probes):
            model.eval()
            probe.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    features = get_layer_features(model, images, layer)
                    outputs = probe(features)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            print(f'Accuracy of probe at layer {layer}: {100 * correct / total}%')

    # Save probes
    os.makedirs(probe_save_path, exist_ok=True)
    for i, probe in enumerate(probes):
        torch.save(probe.state_dict(), os.path.join(probe_save_path, f'probe_{i}.pth'))

    return probes

def calculate_prediction_depth(model, probes, images, device):
    model.eval()
    for probe in probes:
        probe.eval()

    with torch.no_grad():
        final_preds = model(images)[0].argmax(dim=1)
        depths = torch.zeros(images.size(0), dtype=torch.long, device=device)
        for layer, probe in enumerate(probes):
            features = get_layer_features(model, images, layer)
            probe_preds = probe(features).argmax(dim=1)
            depths[(depths == 0) & (probe_preds == final_preds)] = layer + 1

    return depths

def calculate_prediction_depth_with_fix(model, probes, images, device):
    model.eval()
    for probe in probes:
        probe.eval()

    with torch.no_grad():
        final_preds = model(images)[0].argmax(dim=1)
        depths = torch.zeros(images.size(0), dtype=torch.long, device=device)
        for layer, probe in enumerate(probes):
            features = get_layer_features(model, images, layer)
            probe_preds = probe(features).argmax(dim=1)
            depths[(depths == 0) & (probe_preds == final_preds)] = layer + 1

        # Assign the maximum depth to any samples that were not assigned a depth
        depths[depths == 0] = len(probes)

    return depths

def main(args):
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # Set up dataset and model
    rb_ds = eval(f"{args.dataset_name}(\"{args.dataset_path}\")")
    num_classes = rb_ds.num_classes

    model, _ = make_and_restore_model(arch='resnet50', dataset=rb_ds, resume_path=args.model_ckpt_path)
    model = model.to(device)

    if args.train_probes:
        train_loader, val_loader = rb_ds.make_loaders(batch_size=args.batch_size, workers=args.num_workers)
        if args.probe_save_path is None: 
            args.probe_save_path = os.path.dirname(args.model_ckpt_path)
        probes = train_probes_parallel(model, train_loader, val_loader, num_classes, device, args.probe_save_path, num_layers=16, subset_fraction=1.0)
    else:
        # Load probes
        probes = []
        sample_batch, _ = next(iter(rb_ds.make_loaders(batch_size=1, workers=1)[1]))
        sample_batch = sample_batch.to(device)
        for i in range(16):
            with torch.no_grad():
                sample_features = get_layer_features(model, sample_batch, i)
            in_features = sample_features.shape[1]
            probe = LinearProbe(in_features, num_classes).to(device)
            probe.load_state_dict(torch.load(os.path.join(args.probe_save_path, f'probe_{i}.pth')))
            probes.append(probe)

    # Calculate prediction depths
    if args.dirmap_path:
        transform = Compose([Resize(224), CenterCrop((224, 224)), ToTensor()])
        dataset = DirmapDataset(csv_file=args.dirmap_path, transform=transform, dataset_path=args.dataset_path)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate)
    else:
        _, loader = rb_ds.make_loaders(batch_size=args.batch_size, workers=args.num_workers, only_val=True)

    depths = []
    filenames = []

    for batch in loader:
        if args.dirmap_path:
            images, _, idx = batch
            images = images.to(device)
            batch_depths = calculate_prediction_depth_with_fix(model, probes, images, device)
            depths.extend(batch_depths.tolist())
            # filenames.extend(dataset.df.iloc[idx]['im_path'].tolist())
            batch_filenames = [dataset.df.iloc[i]['im_path'] for i in idx]
            filenames.extend(batch_filenames)
        else:
            images, labels = batch
            images = images.to(device)
            batch_depths = calculate_prediction_depth_with_fix(model, probes, images, device)
            depths.extend(batch_depths.tolist())
            filenames.extend([rb_ds.synset_to_filename[rb_ds.label_mapping[l.item()]] for l in labels])

        # Save results to JSON
        results = {fn: depth for fn, depth in zip(filenames, depths)}
        with open(args.json_output_path, 'w') as f:
            json.dump(results, f)

if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    main(args)