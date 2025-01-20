"""Simple way to test models and generate confusion matrices, only works for ResNet50 though. A more feature-rich version is test_model_on_dirmap_get_gt_logit.py
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent)) # Add parent directory to pythonpath
import argparse
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch as ch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop
import seaborn as sns
import matplotlib.pyplot as plt
from utils.dirmap_dataset import DirmapDataset, custom_collate
from utils.mapped_models import imagenet_mapped_model
from robustness.datasets import *
from robustness.model_utils import make_and_restore_model


def get_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dirmap_path', type=str, required=True, help="Path to dirmap csv.")
    parser.add_argument('--dataset_name', type=str, default=None, help="Name of dataset as defined in Robustness library")
    parser.add_argument('--dataset_path', type=str, default="/media/KLAB37/datasets/ImageNet2012", help="Path to dataset")
    parser.add_argument('--model_ckpt_path', type=str, default="/home/morgan/projects/learn-histopath-backend/model_ckpts/imagenet_l2_3_0.pt", help="Path to model checkpoint")
    parser.add_argument('--superclass', type=str, default=None, help="Use superclass. restrictedimagenet | imagenet16")
    parser.add_argument('--num_workers', type=int, default=1, help="Number of CPU threads for dataloader")
    parser.add_argument('--gpu_id', type=int, default=0, help="ID of GPU to use")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")

    if argv is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(argv)

    return vars(args)

def test_model_accuracy_and_generate_confusion_matrix(**kwargs):
    transform = Compose([Resize(224), CenterCrop((224, 224)), ToTensor()])
    dataset = DirmapDataset(csv_file=kwargs['dirmap_path'], transform=transform, class_num_col="orig_class_num")
    loader = DataLoader(dataset, batch_size=kwargs['batch_size'], num_workers=kwargs['num_workers'], collate_fn=custom_collate)

    rb_ds = eval(f"{kwargs['dataset_name']}(\"{kwargs['dataset_path']}\")")
    model, _ = make_and_restore_model(arch='resnet50', dataset=rb_ds, gpu_id=kwargs['gpu_id'], resume_path=kwargs['model_ckpt_path'])
    model.cuda(kwargs['gpu_id'])
    model.eval()

    all_preds = []
    all_labels = []

    with ch.no_grad():
        for data in tqdm(loader):

            if kwargs['dirmap_path']:
              im, label, _ = data
            else:
              im, label = data

            im = im.cuda(kwargs['gpu_id'])
            outputs, _ = model(im, make_adv=False)
            _, preds = ch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    print(f"Accuracy: {accuracy}")

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")

    return accuracy, cm

if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    test_model_accuracy_and_generate_confusion_matrix(**args)
