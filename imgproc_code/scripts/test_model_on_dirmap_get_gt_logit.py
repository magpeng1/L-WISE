"""Evaluate a trained model on an image dataset, and calculate the logit for the ground truth class on each image (this can be used as a difficulty score)
This script will save a .csv file at the same location as the original .csv file that lists the images, unless a --suffix is specified to make it a different file name. 
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop
import seaborn as sns
import matplotlib.pyplot as plt
from utils.dirmap_dataset import DirmapDataset, custom_collate
from utils.mapped_models import imagenet_mapped_model
from robustness.datasets import *
from robustness.model_utils import make_and_restore_model
from robustness.imagenet_models import *
import timm
import dill


def get_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dirmap_path', type=str, required=True, help="Path to dirmap csv. PLEASE NOTE: will be overwritten unless --suffix is specified")
    parser.add_argument('--suffix', type=str, default="", help="Something to add to the end of the new .csv file name")
    parser.add_argument('--dataset_name', type=str, default=None, required=True, help="Name of dataset as defined in Robustness library")
    parser.add_argument('--dataset_path', type=str, default="/media/KLAB37/datasets/ImageNet2012", help="Path to dataset")
    parser.add_argument('--arch', type=str, default='resnet50', help="Name of CNN archhitecture (e.g. resnet50, convnext_tiny, densenet201)")
    parser.add_argument('--model_ckpt_path', type=str, default="model_ckpts/imagenet_l2_3_0.pt", help="Path to model checkpoint")
    parser.add_argument('--superclass', type=str, default=None, help="Use superclasses to calculate difficulty. restrictedimagenet | imagenet16")
    parser.add_argument('--num_workers', type=int, default=1, help="Number of CPU threads for dataloader")
    parser.add_argument('--gpu_id', type=int, default=0, help="ID of GPU to use")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--split', type=str, default=None, help="Evaluate only one split from the dirmap. train | test | val")
    parser.add_argument('--confusion_matrix', default=False, action='store_true', help="generate a confusion matrix")

    parser.add_argument('--class_num_col', type=str, default="class_num", help="Which column of the dirmap to use to determine the class number. Might need to be set to class_num")
    parser.add_argument('--gt_logit_col_name', type=str, default="robust_gt_logit", help="Custom name for column where ground truth logit will be saved.")

    parser.add_argument('--vanilla', default=False, action='store_true', help="save gt logit with column name 'vanilla_gt_logit' instead of 'robust_gt_logit'. Does nothing except change column names to 'vanilla' for convenience.")
    parser.add_argument('--pytorch_pretrained', default=False, action='store_true', help="Use a pytorch-pretrained model (usually used with --vanilla)")

    parser.add_argument('--entropy', default=False, action='store_true', help="Save the entropy in addition to the robust ground truth logit")
    parser.add_argument('--cross_entropy', default=False, action='store_true', help="Save the cross-entropy in addition to the robust ground truth logit")

    parser.add_argument('--present_classes_only', action='store_true', help="Evaluate accuracy only among classes present in the dataset")

    if argv is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(argv)

    return vars(args)


def test_model_accuracy_and_generate_confusion_matrix(**kwargs):
    transform = Compose([Resize(224), CenterCrop((224, 224)), ToTensor()])
    dataset = DirmapDataset(csv_file=kwargs['dirmap_path'], relative_path=True, transform=transform, class_num_col=kwargs['class_num_col'], dataset_path=kwargs["dataset_path"], split=kwargs['split'], use_df_idx=True)
    loader = DataLoader(dataset, batch_size=kwargs['batch_size'], num_workers=kwargs['num_workers'], collate_fn=custom_collate)

    rb_ds = eval(f"{kwargs['dataset_name']}(\"{kwargs['dataset_path']}\")")
    if kwargs['pytorch_pretrained']:
        model, _ = make_and_restore_model(arch=kwargs['arch'], dataset=rb_ds, gpu_id=kwargs['gpu_id'], resume_path=None, pytorch_pretrained=True)
    else:
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
        elif 'vit' in kwargs['arch']:
            num_classes = 1000 if ('ImageNet' in kwargs['dataset_name']) else rb_ds.num_classes
            vit_model = vit.create_vit_model(num_classes=num_classes, pretrained=False, from_tf=(kwargs['arch'] == 'harmonized_vit')) # Harmonized ViT is converted from TensorFlow
            model, _ = make_and_restore_model(arch=vit_model, dataset=rb_ds, gpu_id=kwargs['gpu_id'], resume_path=kwargs['model_ckpt_path'])
        elif '_external_checkpoint' in kwargs['arch']:
            init_model = rb_ds.get_model(kwargs['arch'].split("_external_checkpoint")[0], pretrained=False)
            state_dict = ch.load(kwargs['model_ckpt_path'])
            state_dict = {k[len('module.'):]:v for k,v in state_dict['state_dict'].items()}
            init_model.load_state_dict(state_dict)
            model, _ = make_and_restore_model(arch=init_model, dataset=rb_ds, gpu_id=kwargs['gpu_id'])
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

    all_preds = []
    all_labels = []

    dirmap = pd.read_csv(kwargs['dirmap_path'])
    logit_col_name = 'vanilla_gt_logit' if kwargs['vanilla'] else kwargs['gt_logit_col_name']
    entropy_col_name = 'vanilla_entropy' if kwargs['vanilla'] else 'robust_entropy'
    cross_entropy_col_name = 'vanilla_cross_entropy' if kwargs['vanilla'] else 'robust_cross_entropy'
    if logit_col_name not in dirmap.columns:
        dirmap[logit_col_name] = np.nan

    # Get the present classes
    present_classes = dataset.df[kwargs['class_num_col']].unique()
    present_classes.sort()
    class_to_idx = {cls: idx for idx, cls in enumerate(present_classes)}

    with ch.no_grad():
        for data in tqdm(loader):
            if kwargs['dirmap_path']:
                im, label, idx = data
            else:
                im, label = data

            im = im.cuda(kwargs['gpu_id'])
            outputs, _ = model(im, make_adv=False)

            if kwargs['present_classes_only']:
                # Filter outputs to only include present classes
                filtered_outputs = outputs[:, present_classes]
                _, preds = ch.max(filtered_outputs, 1)
                # Map predictions back to original class numbers
                preds = ch.tensor([present_classes[p.item()] for p in preds])
            else:
                _, preds = ch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.numpy())

            # Calculate entropy and cross-entropy
            if kwargs['entropy']:
                if kwargs['present_classes_only']:
                    log_probs = F.log_softmax(filtered_outputs, dim=1)
                    probs = F.softmax(filtered_outputs, dim=1)
                else:
                    log_probs = F.log_softmax(outputs, dim=1)
                    probs = F.softmax(outputs, dim=1)

                entropy = -(probs * log_probs).sum(dim=1)
            
            if kwargs['cross_entropy']:
                if kwargs['present_classes_only']:
                    label_indices = ch.tensor([class_to_idx[l.item()] for l in label]).cuda(kwargs['gpu_id'])
                    cross_entropy = F.cross_entropy(filtered_outputs, label_indices, reduction='none')
                else:
                    cross_entropy = F.cross_entropy(outputs, label.cuda(kwargs['gpu_id']), reduction='none')

            for j, output in enumerate(outputs):
                dirmap.at[idx[j], logit_col_name] = output[label[j]].item()
                if kwargs['entropy']:
                    dirmap.at[idx[j], entropy_col_name] = entropy[j].item()
                if kwargs['cross_entropy']:
                    dirmap.at[idx[j], cross_entropy_col_name] = cross_entropy[j].item()

    accuracy = accuracy_score(all_labels, all_preds)

    cm = None
    if kwargs['confusion_matrix']:
        cm = confusion_matrix(all_labels, all_preds)
        class_names = dataset.df['class'].unique()

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.savefig("confusion_matrix.png")

    base_csv_path, ext = os.path.splitext(kwargs['dirmap_path'])
    suffix = kwargs['suffix']
    if len(suffix) > 0 and suffix[0] != "_":
        suffix = "_" + suffix
    csv_save_path = base_csv_path + suffix + ext
    dirmap.to_csv(csv_save_path, index=False)

    return accuracy, cm


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    accuracy, cm = test_model_accuracy_and_generate_confusion_matrix(**args)
    print("Accuracy:", accuracy)