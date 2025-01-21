# L-WISE: Boosting Human Image Category Learning Through Model-Based Image Selection and Enhancement

## Abstract
The currently leading artificial neural network (ANN) models of the visual ventral stream -- which are derived from a combination of performance optimization and robustification methods -– have demonstrated a remarkable degree of behavioral alignment with humans on visual categorization tasks. Extending upon previous work, we show that not only can these models guide image perturbations that change the induced human category percepts, but they also can enhance human ability to accurately report the original ground truth. Furthermore, we find that the same models can also be used out-of-the-box to predict the proportion of correct human responses to individual images, providing a simple, human-aligned estimator of the relative difficulty of each image. Motivated by these observations, we propose to augment visual learning in humans in a way that improves human categorization accuracy at test time. Our learning augmentation approach consists of (i) selecting images based on their model-estimated recognition difficulty, and (ii) using image perturbations that aid recognition for novice learners. We find that combining these model-based strategies gives rise to test-time categorization accuracy gains of 33-72% relative to control subjects without these interventions, despite using the same number of training feedback trials. Surprisingly, beyond the accuracy gain, the training time for the augmented learning group was also shorter by 20-23%. We demonstrate the efficacy of our approach in a fine-grained categorization task with natural images, as well as tasks in two clinically relevant image domains -- histology and dermoscopy -- where visual learning is notoriously challenging. To the best of our knowledge, this is the first application of ANNs to increase visual learning performance in humans by enhancing category-specific features.

## How to enhance images using robust networks

### Method 1: Jupyter notebook

@@@@ TODO

### Method 2: python script to enhance a dataset

The script imgproc_code/scripts/enhance.py is designed to "enhance" an entire dataset of images indexed using a "dirmap" csv file (see [Dataset organization section](#dataset-organization-and-how-to-add-new-datasets-to-this-project)) using a robust model. A GPU is required for execution. Please see the arguments You must provide, at minimum:
* --dest_dir (path to a location where the the enhanced version of the dataset will be saved)
* --dirmap_path (path to a csv file, with one row for each image to be enhanced - see see [Dataset organization section](#dataset-organization-and-how-to-add-new-datasets-to-this-project) for formatting)
* --dataset_name (name of the dataset class in imgproc_code/robustness/robustness/datasets.py)
* --model_ckpt_path (path to the trained model that will be used to enhance the images)
* --arch (name of CNN architecture. As of now, 'resnet50' works reliably and all other options are experimental.)

Other optional arguments can be used to set a specific L2 pixel budget for the perturbations (--eps), which loss function should be used (--objective_type), and many other aspects of the enhancement process - see script arguments for details. If you run out of GPU memory, try reducing the --batch_size. Here is an example terminal command to enhance some imagenet images using a pretrained model (run from the imgproc_code directory):
```
python enhance.py --eps 20 --num_steps 40 --dest_dir data/enhanced_imagenet_images --dirmap_path path/to/dirmap.csv --dataset_name ImageNet --dataset_path path/to/ImageNet --model_ckpt_path model_ckpts/ImageNet_eps3.pt --objective_type logit
```

Our enhancement approach essentially involves maximizing the logit value of the ground truth class. You can also minimize the cross-entropy loss by setting --objective_type cross_entropy. For the fine-grained datasets we used in the learning experiments, we used --objective_type logit_diverge, such that the logits of competing classes are explicitly minimized - this seems to produce more compelling perturbations in fine-grained tasks. For example, to enhance HAM10000 dermoscopy images:
```
python enhance.py --eps 8 --num_steps 16 --dest_dir data/enhanced_dermoscopy_images --dirmap_path path/to/dirmap.csv --dataset_name HAM10000 --dataset_path path/to/HAM10000 --model_ckpt_path model_ckpts/HAM10000_eps1.pt --objective_type logit_diverge
```

The additional script imgproc_code/scripts/enhance_vit_aug.py is similar to enhance.py, but it is designed specifically to use a vision transformer model called XCiT and implements an array of multi-view augmentations to generate gradient steps for higher-quality perturbations (which seems to be necessary specifically for transformer-based models.) Here is an example terminal command to run this script on some ImageNet animal images (if you run out of GPU memory, try reducing the batch size): 
```
python scripts/enhance_vit_aug.py --dest_dir data/imagenet16_xcit_tuned --eps 20 --step_size 0.5 --num_steps 80 --num_augs 10 --batch_size 8 --dirmap_path data/imagenet16/dirmap.csv --objective_type logit --save_originals --dataset_name ImageNet --dataset_path data/imagenet16 --arch xcit_large_12_p16_224 --model_ckpt_path model_ckpts/debenedetti/xcit-l12-ImageNet-eps-4.pth.tar
```

### Method 3: bash scripts for enhancing datasets and uploading to S3 with multiple perturbation sizes

We provide several bash scripts that automate the process of enhancing entire datasets with multiple different perturbation sizes (pixel budget values "epsilon"), and uploading the resulting copies of the dataset to S3 to be used in psychophysics experiments. They are found in the imgproc_code/scripts/batch_enhance directory. Some modification of the dataset/csv/model checkpoint paths will be requried to get these scripts working on your system. 

## How to predict difficulty of images using robust networks

Our difficulty prediction metric is the logit value from a robust CNN associated with the groundtruth class. This can be calculated for images from many datasets using imgproc_code/scripts/test_model_on_dirmap_get_gt_logit.py (which also serves as a way to evaluate a trained network on a dataset). 

Example terminal commands (to be run from inside imgproc_code, after obtaining/setting up these datasets and downloading model checkpoints): 

```
# Get logits for ImageNet Animals (16 classes):
python test_model_on_dirmap_get_gt_logit.py --dirmap_path path/to/dataset_dirmap.csv --dataset_name ImageNet --model_ckpt_path model_ckpts/ImageNet_eps3.pt --class_num_col orig_class_num

# Get logits for "Idaea4" moth photos (4 classes), and also generate a class confusion matrix:
python test_model_on_dirmap_get_gt_logit.py --dirmap_path data/idaea4/idaea4_natural/dirmap.csv --dataset_name idaea4 --dataset_path path/to/idaea4_natural --model_ckpt_path model_ckpts/idaea4_eps1.pt --confusion_matrix
```
Note that "--class_num_col orig_class_num" is specified for ImageNet so that we evaluate the ground truth logits on the original 1000 classes, not the superclasses (i.e., how confident is the model that a specific image is "Siberian Husky" rather than "dog" in general)


## Dataset organization, and how to add new datasets to this project

In order to add new image classification datasets to this project, there are three requirements: 
1. Datasets must be indexed using "dirmap" csv files
2. (for model training/evaluation only) Datasets must be arranged in "[ImageFolder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html) format" (root dataset folder contains split folders like "train", "val", "test", within each of these is one folder per class, containing all images of that class within that split)
3. (for model training/evaluation only) A new class for each dataset must (in most cases) be added to imgproc_code/robustness/robustness/datasets.py 

### "Dirmap" indexing for datasets

In this codebase, datasets are generally organized using "dirmap" csv files for each dataset (often imported as Pandas dataframes). A dirmap contains one row per image, with several important columns that define image attributes:
* split (set to "train", "val", or "test")
* class (string name of the image's class)
* class_num (integer ID associated with the image's class)
* im_path (path to the image file, relative to the directory the dirmap csv file is placed in. e.g. "val/dog/ILSVRC2012_val_00000269.JPEG")

Optional columns, depending on what the dirmap's origin and what it is being used for, include: 
* orig_class (string name of the original class before reassignment - e.g., in ImageNet Animals, class=dog when orig_class=Siberian_husky)
* orig_class_num (integer ID associated with the original class)
* orig_im_path (relative path to the image file before dataset reorganization)
* url (url to access the image online, often from an S3 bucket. A script for uploading image datasets to S3, while adding this column to the dirmap, can be found at imgproc_code/scripts/upload_images_s3)
* robust_gt_logit (Ground truth logit from a robust model, used to calculate difficulty. Can be added using imgproc_code/scripts/test_model_on_dirmap_get_gt_logit.py)
* difficulty (A difficulty score, which might, for example, be a normalized/transformed version of robust_gt_logit)
* Additional dataset-specific columns (e.g., the MHIST dataset's dirmap has a column "Number of Annotators who Selected SSA (Out of 7)" indicating expert agreement on the image labels).

Dirmap files are initially produced using scripts placed in imgproc_code/dataset_setup, such as define_imagenet_subset_dataset.py (for ImageNet Animals, use 'i16' option), define_ham10000_dataset.py (for HAM10000 dermoscopy dataset), etc. Some of the scripts will work for multiple datasets - for example, imgproc_code/dataset_setup/imagefolder_style_df.py will produce a valid dirmap for any dataset that is already in "[ImageFolder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html) format."

In order to train/evaluate models on image datasets within the [Robustness library](https://github.com/MadryLab/robustness) (which this repository builds upon, and contains a modified version of), datasets must be organized in "[ImageFolder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html) format" (root dataset folder contains split folders like "train", "val", "test", within each of these is one folder per class, containing all images of that class within that split). The script **imgproc_code/dataset_setup/build_dataset.py** is designed to create ImageFolder-formatted copies of datasets that do not start out with this format, using an initial "recipe" dirmap csv produced by other scripts in imgproc_code/dataset_setup. It must be given the path to the recipe dirmap csv, as well as the path to the dataset root. It also has other features, such as the ability to sample a class-balanced subset of the dataset, resize/reformat images, and select a subset of classes to include - see arguments list in the script for details. 

### Setting up new datasets for model training/evaluation

To train robust models on new, outside datasets, you must define a new class in imgproc_code/robustness/robustness/datasets.py. You can also add hyperparameter defaults in imgproc_code/robustness/robustness/defaults.py, and custom data augmentations in imgproc_code/robustness/robustness/data_augmentation.py. See also the original [Robustness library documentation](https://robustness.readthedocs.io/en/latest/example_usage/training_lib_part_2.html#training-on-custom-datasets) (this project includes a modified copy of the library)


## How to train/fine-tune robust models

This project builds directly on top of the [Robustness library](https://github.com/MadryLab/robustness) by the Madry Lab - if you simply wish to experiment with adversarially trained models, you may be better off using that library directly. Our code provides additional functionality for fine-tuning pretrained models through adversarial training.

You can fine-tune an existing model using adversarial training via the script **imgproc_code/scripts/robust_transfer_learning.py**. Please review the arguments list of this script to understand how to use it. Note that, before training/fine-tuning a model on a new, oustide dataset, you must format the dataset appropriately (see [Dataset organization, and how to add new datasets to this project](#dataset-organization-and-how-to-add-new-datasets-to-this-project)).

For example, here is how to adversarially-fine-tune an adversarially-ImageNet-pretrained model on the MHIST histology dataset (with an adversarial epsilon of 1 during fine-tuning):
```
python scripts/robust_transfer_learning.py --dataset_name MHIST --dataset_path path/to/imagefolder/formatted/mhist --n_epochs 50 --lr 0.001 --custom_lr_multiplier "" --batch_size 16 --eps 1 --saved_model_ckpt_path model_ckpts/ImageNet_eps3.pt
```

You can also adversarially train models from scratch using this same script. For example, here is how to replicate our adversarially training run of a ResNet50 model on the iNaturalist dataset from scratch (run from inside imgproc_code directory):
```
# In one shot (may take a few weeks)
python scripts/robust_transfer_learning.py --eps 1 --attack_steps 7 --attack_lr 0.3 --n_epochs 200 --lr 0.1 --step_lr 50 --step_lr_gamma 0.1 --gpu_ids 0 --custom_lr_multiplier "" --batch_size 256 --val_batch_size 128 --n_workers 16 --dataset_name inat --dataset_path path/to/inat2021

# In two subsequent jobs (For the second job, change "85ad31ec-6919-52878a26a9f8" to the output directory of the first job)
python scripts/robust_transfer_learning.py --eps 1 --attack_steps 7 --attack_lr 0.3 --n_epochs 100 --lr 0.1 --step_lr 50 --step_lr_gamma 0.1 --gpu_ids 0 --custom_lr_multiplier "" --batch_size 256 --val_batch_size 128 --n_workers 16 --dataset_name inat --dataset_path path/to/inat2021

python scripts/robust_transfer_learning.py --eps 1 --attack_steps 7 --attack_lr 0.3 --n_epochs 100 --lr 0.001 --step_lr 50 --step_lr_gamma 0.1 --gpu_ids 0 --custom_lr_multiplier "" --batch_size 256 --val_batch_size 128 --n_workers 16 --dataset_name inat --dataset_path path/to/inat2021 --continue_same_dataset --saved_model_ckpt_path train_output/85ad31ec-6919-52878a26a9f8/checkpoint.pt.latest
```
Note here we train with an adversarial epsilon of 1 (7 attack steps with step size 0.3), for 200 epochs, with a starting learning rate of 0.1, and multiplying the learning rate by a factor of 0.1 (--step_lr_gamma) every 50 epochs (--step_lr). 

## Running psychophysics experiments

@@@@ TODO
