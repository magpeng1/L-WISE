import os
import sys
import torch
import argparse
from robustness import model_utils, datasets, train, defaults
from robustness.datasets import *
from torch import nn
from timm.data import Mixup
from cox.utils import Parameters
import cox.store


def get_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, default="AcevedoWBC", help="Name of dataset (see robustness/datasets.py)")
    parser.add_argument('--dataset_path', type=str, default="/media/KLAB37/morgan_data/wbc/acevedo_wbc/robustness", help="Path to dataset")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--step_lr', type=int, default=None, help="Number of epochs after which to decrease learning rate (must use --custom_lr_multiplier "")")
    parser.add_argument('--step_lr_gamma', type=float, default=None, help="Factor by which to update the learning rate at each lr schedule step (e.g., 0.1 decreases it by a factor of 10)")
    parser.add_argument('--weight_decay', type=float, default=None, help="Weight decay for SGD optimizer")
    parser.add_argument('--custom_lr_multiplier', type=str, default="", help="Learning rate schedule (cyclic | linear | cosine)")
    parser.add_argument('--lr_interpolation', type=str, default='step', help="How to drop the learning rate, either ``step`` or ``linear``, ignored unless ``custom_lr_multiplier`` is provided.")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--val_batch_size', type=int, default=64, help="Evaluation batch size")
    parser.add_argument('--n_epochs', type=int, default=50, help="Number of epochs")
    parser.add_argument('--eps', type=float, default=3, help="Epsilon budget for adversarial training")
    parser.add_argument('--attack_lr', type=float, default=0.5, help="Learning rate for adversarial attack steps")
    parser.add_argument('--attack_steps', type=int, default=7, help="Number of attack steps for adversarial training")
    parser.add_argument('--use_mixup_and_cutmix', default=False, action='store_true', help="Use cutmix and mixup in the data augmentation pipeline")
    # parser.add_argument('--gpu_id', type=int, default=0, help="GPU id (integer)")
    parser.add_argument('--gpu_ids', nargs="+", type=int, default=[0, 1], help="GPU id (integer)")
    parser.add_argument('--n_workers', type=int, default=8, help="Number of dataloader worker threads")
    parser.add_argument('--arch', type=str, default='resnet50', help="Name of CNN archhitecture (e.g. resnet50, convnext_tiny, densenet201)")
    parser.add_argument('--saved_model_ckpt_path', type=str, default=None, help="Path to saved model ckpt")
    parser.add_argument('--pytorch_pretrained', default=False, action='store_true',
                        help="Use a pytorch pretrained model (only applicable if no ckpt path provided)")
    parser.add_argument('--output_path', type=str, default="train_output", help="Path to save results")
    parser.add_argument('--not_robust', default=False, action='store_true',
        help="If this is FALSE (default), do adversarial robustness training. If true, do regular non-robust training.")
    parser.add_argument('--continue_same_dataset', default=False, action='store_true',
        help="Add this argument when you are continuing training from a checkpoint with the same number of output nodes as the dataset you'll be using")
    parser.add_argument('--pretraining_dataset_name', default="ImageNet", help="Name of the dataset that was used to pretrain the model (allows it to initialize with right number of classes)")
    parser.add_argument('--last_layer_only', action='store_true', default=False, help="If set, freeze all layers except the final output layer")
    parser.add_argument('--freeze_first_k_conv_layers', type=int, default=0, help="Freeze the first k convolutional layers (0 means no freezing)")
    parser.add_argument('--save_every_epoch', default=False, action='store_true', help="Save a .ckpt for every epoch of training")
    parser.add_argument('--eval', default=False, action='store_true', help="Separately evaluate the model's performance at the end")
    parser.add_argument('--eval_only', default=False, action='store_true', help="Do not train, and ONLY evaluate the model's performance")
    parser.add_argument('--verbose', default=False, action='store_true', help="Verbose mode: print out all losses")

    return parser.parse_args(argv)


args = get_args(sys.argv[1:])

if args.gpu_ids:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpu_ids))
    gpu_ids = list(range(len(args.gpu_ids)))
else:
    gpu_ids = [0]

out_store = cox.store.Store(args.output_path)

# Load dataset
dataset = eval(f"{args.dataset_name}(\"{args.dataset_path}\")")

train_loader, val_loader = dataset.make_loaders(batch_size=args.batch_size, val_batch_size=args.val_batch_size, workers=args.n_workers)

# Model
if args.saved_model_ckpt_path is None:
    if args.pytorch_pretrained:
        # First load model with ImageNet (matched # of output nodes), then change it to have the appropriate number of classes.
        model, _ = model_utils.make_and_restore_model(arch=args.arch, dataset=ImageNet("./data/"), pytorch_pretrained=True)
        if args.arch == 'resnet50':
            model.model.fc = nn.Linear(in_features=2048, out_features=dataset.num_classes, bias=True)
        elif 'convnext' in args.arch.lower():
            in_features = model.model.model.classifier[2].in_features
            model.model.model.classifier[2] = nn.Linear(in_features, dataset.num_classes)
        elif 'densenet' in args.arch.lower():
            in_features = model.model.classifier.in_features
            model.model.classifier = nn.Linear(in_features, dataset.num_classes)
        else:
            raise NotImplementedError()
    else:
        model, _ = model_utils.make_and_restore_model(arch=args.arch, dataset=dataset)
else:
    if args.continue_same_dataset:
        model, _ = model_utils.make_and_restore_model(arch=args.arch, dataset=dataset, resume_path=args.saved_model_ckpt_path)
    else:
        # First load model with ImageNet (matched # of output nodes), then change it to have the appropriate number of classes.
        model, _ = model_utils.make_and_restore_model(arch=args.arch, dataset=eval(f"{args.pretraining_dataset_name}(\"./data/\")"), resume_path=args.saved_model_ckpt_path)
        print("Num classes: " + str(dataset.num_classes))
        if args.arch == 'resnet50':
            model.model.fc = nn.Linear(in_features=2048, out_features=dataset.num_classes, bias=True)
        elif 'convnext' in args.arch.lower():
            in_features = model.model.model.classifier[2].in_features
            model.model.model.classifier[2] = nn.Linear(in_features, dataset.num_classes)
        elif 'densenet' in args.arch.lower():
            in_features = model.model.classifier.in_features
            model.model.classifier = nn.Linear(in_features, dataset.num_classes)
        else:
            raise NotImplementedError()
        model.num_classes = dataset.num_classes

def freeze_all_except_last_layer(model):
    for name, param in model.named_parameters():
        if "fc" not in name:  # Assuming the last layer is named "fc"
            param.requires_grad = False
    return model

def freeze_first_k_conv_layers(model, k):
    conv_layer_count = 0
    for name, param in model.named_parameters():
        if 'conv' in name:
            if conv_layer_count < k:
                param.requires_grad = False
                conv_layer_count += 1
            else:
                break
    return model, conv_layer_count

if args.last_layer_only:
    model = freeze_all_except_last_layer(model)
    print("All layers except the last have been frozen.")
elif args.freeze_first_k_conv_layers > 0:
    model, frozen_layers = freeze_first_k_conv_layers(model, args.freeze_first_k_conv_layers)
    print(f"The first {frozen_layers} convolutional layers have been frozen.")

# Hard-coded base parameters
train_kwargs = {
    'out_dir': "train_out",
    'adv_train': not args.not_robust, # Will be "True" unless --not_robust is specified
    'constraint': '2',
    'eps': args.eps,
    'attack_lr': args.attack_lr,
    'attack_steps': args.attack_steps,
    'epochs': args.n_epochs,
    'custom_lr_multiplier': args.custom_lr_multiplier,
    'lr_interpolation': args.lr_interpolation,
}
if args.lr is not None:
    train_kwargs['lr'] = args.lr
if args.step_lr is not None: 
    train_kwargs['step_lr'] = args.step_lr
if args.step_lr_gamma is not None:
    train_kwargs['step_lr_gamma'] = args.step_lr_gamma
if args.weight_decay is not None: 
    train_kwargs['weight_decay'] = args.weight_decay
if args.save_every_epoch:
    train_kwargs['save_ckpt_iters'] = 1
train_kwargs['use_mixup_and_cutmix'] = args.use_mixup_and_cutmix

train_args = Parameters(train_kwargs)

# Fill whatever parameters are missing from the defaults
train_args = defaults.check_and_fill_args(train_args,
                        defaults.TRAINING_ARGS, ImageNet)
train_args = defaults.check_and_fill_args(train_args,
                        defaults.PGD_ARGS, ImageNet)

print("Training arguments:")
print(train_args)

# primary_gpu = args.gpu_ids[0] if args.gpu_ids else 0
# device = torch.device(f"cuda:{primary_gpu}" if torch.cuda.is_available() else "cpu")
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# torch.distributed.init_process_group()

# primary_gpu = gpu_ids[0] if gpu_ids else 0
torch.cuda.set_device(gpu_ids[0])

if args.use_mixup_and_cutmix:
    train_args.mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None, prob=0.3, switch_prob=0.5, mode='batch', label_smoothing=0, num_classes=dataset.num_classes)

if not args.eval_only:
    train.train_model(train_args, model, (train_loader, val_loader), store=out_store, dp_device_ids=gpu_ids, verbose=args.verbose)

if args.eval or args.eval_only:
    class EvalArgs:
        def __init__(self):
            self.adv_eval = not args.not_robust
            self.adv_train = not args.not_robust
            self.attack_steps = args.attack_steps
            self.attack_lr = args.attack_lr
            self.eps = args.eps
            self.use_best = True
            self.random_restarts = False
            self.constraint = '2'
            self.batch_size = args.batch_size

    eval_results = train.eval_model(EvalArgs(), model, val_loader, None)
    print("MODEL EVALUATION RESULTS:")
    print(eval_results)
