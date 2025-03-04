# L-WISE: Boosting Human Visual Category Learning Through Model-Based Image Selection and Enhancement (ICLR 2025)

[Link to project website](https://morganbdt.github.io/L-WISE/) | [Link to preprint on arXiv](https://arxiv.org/pdf/2412.09765)

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
## Table of Contents

  - [Codebase Overview and Setup](#codebase-overview-and-setup)
- [Enhancing images, predicting image difficulty, and training robust models (see "imgproc_code" directory)](#enhancing-images-predicting-image-difficulty-and-training-robust-models-see-imgproc_code-directory)
  - [Obtaining pretrained model checkpoints](#obtaining-pretrained-model-checkpoints)
  - [How to enhance images using robust networks](#how-to-enhance-images-using-robust-networks)
    - [Method 1: python script to enhance a dataset](#method-1-python-script-to-enhance-a-dataset)
    - [Method 2: bash scripts for enhancing datasets and uploading to S3 with multiple perturbation sizes](#method-2-bash-scripts-for-enhancing-datasets-and-uploading-to-s3-with-multiple-perturbation-sizes)
  - [How to predict difficulty of images using robust networks](#how-to-predict-difficulty-of-images-using-robust-networks)
  - [How to train/fine-tune robust models](#how-to-trainfine-tune-robust-models)
  - [Dataset organization, and how to add new datasets to this project](#dataset-organization-and-how-to-add-new-datasets-to-this-project)
    - ["Dirmap" indexing for datasets](#dirmap-indexing-for-datasets)
    - [Setting up new datasets for model training/evaluation](#setting-up-new-datasets-for-model-trainingevaluation)
- [Running psychophysics experiments (see "psych_code" directory)](#running-psychophysics-experiments-see-psych_code-directory)
  - [Quick start: hosting pre-designed experiments from the L-WISE paper on Prolific and AWS](#quick-start-hosting-pre-designed-experiments-from-the-l-wise-paper-on-prolific-and-aws)
    - [How to deploy experiments to reproduce L-WISE paper results:](#how-to-deploy-experiments-to-reproduce-l-wise-paper-results)
    - [Tools for hosting experiments on Mechanical Turk](#tools-for-hosting-experiments-on-mechanical-turk)
  - [Obtaining and analyzing public data from the original L-WISE experiments](#obtaining-and-analyzing-public-data-from-the-original-l-wise-experiments)
  - [Subdirectories in experiment_files (one for each experiment)](#subdirectories-in-experiment_files-one-for-each-experiment)
    - [**Descriptions of experiment_files directories:**](#descriptions-of-experiment_files-directories)
  - [HTML files for experimental code](#html-files-for-experimental-code)
  - [Dataset_dirmap.csv and trialsets.csv](#dataset_dirmapcsv-and-trialsetscsv)
  - [Configuring the experiments using config.yaml](#configuring-the-experiments-using-configyaml)
    - [Global and trial-block-specific settings](#global-and-trial-block-specific-settings)
    - [Class/choice names and aliases](#classchoice-names-and-aliases)
    - [Keyboard control for binary classification tasks](#keyboard-control-for-binary-classification-tasks)
    - [Specifying different trial types](#specifying-different-trial-types)
  - [Deploying experiments to online platforms](#deploying-experiments-to-online-platforms)
    - [Initial deployment to AWS](#initial-deployment-to-aws)
    - [Downloading data from deployed experiments](#downloading-data-from-deployed-experiments)
  - [Key steps when implementing a new experiment](#key-steps-when-implementing-a-new-experiment)
  - [Implementing experiments with customized trial sequences](#implementing-experiments-with-customized-trial-sequences)
  - [Acknowledgements](#acknowledgements)
  - [Citation](#citation)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Codebase Overview and Setup

The codebase for this project is divided into two main parts. The **imgproc_code** directory contains code for enhancing images, predicting image difficulty, training models, and various other image processing tasks. The **psych_code** directory contains customizable code for running psychophysics experiments. 

The following shell commands can be used to set up a suitable Python virtual environment (please run from the root project directory, i.e. "L-WISE"):

```
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev

mkdir -p venvs
python3.12 -m venv ./venvs/lwise_env
source venvs/lwise_env/bin/activate
pip install -r requirements.txt
pip install -e imgproc_code/robustness
pip install -e imgproc_code/lwise_imgproc_utils
pip install -e psych_code
```

# Enhancing images, predicting image difficulty, and training robust models (see "imgproc_code" directory)

## Obtaining pretrained model checkpoints

Run the following commands from the root project directory to download and extract the ResNet-50 model checkpoints: 
```
mkdir -p imgproc_code/model_ckpts
wget https://github.com/MorganBDT/L-WISE/releases/download/v1.0.0/L-Wise_ckpts.zip
unzip L-Wise_ckpts.zip -d imgproc_code/model_ckpts
rm L-Wise_ckpts.zip
```

We provide adversarially-trained/robustified models for all of the datasets we worked with, along with "vanilla" (non-robust) versions otherwise trained with the same hyperparameters. The name of each .pt model file indicates the dataset it was trained or fine-tuned on, as well as the training ϵ for the robust models (e.g., ϵ=1 or ϵ=3).

For some of the supplementary experiments in the L-WISE paper, we also used pretrained checkpoints from [Debenedetti et al.](https://github.com/dedeswim/vits-robustness-torch) (XCiT-L12 adversarially pretrained on ImageNet with ϵ=4), [Yun et al.](https://github.com/clovaai/CutMix-PyTorch) (ResNet-50 pretrained on ImageNet with CutMix), and [Gaziv et al.](https://github.com/ggaziv/Wormholes) (ResNet-50 models adversarially pretrained on ImageNet with several different ϵ values).

## How to enhance images using robust networks

### Method 1: python script to enhance a dataset

The script imgproc_code/scripts/enhance.py is designed to "enhance" an entire dataset of images indexed using a "dirmap" csv file (see [Dataset organization section](#dataset-organization-and-how-to-add-new-datasets-to-this-project)) using a robust model. A GPU is required for execution. Please see the arguments in the script for details. You must provide, at minimum:
* --dest_dir &emsp; (path to a location where the the enhanced version of the dataset will be saved)
* --dirmap_path &emsp; (path to a csv file, with one row for each image to be enhanced - see [Dataset organization section](#dataset-organization-and-how-to-add-new-datasets-to-this-project) for formatting)
* --dataset_name &emsp; (name of the dataset class in imgproc_code/robustness/robustness/datasets.py)
* --model_ckpt_path &emsp; (path to the trained model that will be used to enhance the images)
* --arch &emsp; (name of CNN architecture. As of now, 'resnet50' works reliably and all other options are experimental.)

Other optional arguments can be used to set a specific L2 pixel budget for the perturbations (--eps), which loss function should be used (--objective_type), and many other aspects of the enhancement process - see script arguments for details. If you run out of GPU memory, try reducing the --batch_size. Here is an example terminal command to enhance some imagenet images using a pretrained model (run from the imgproc_code directory):
```
python scripts/enhance.py --eps 20 --num_steps 40 --dest_dir data/enhanced_imagenet_images --dirmap_path path/to/dirmap.csv --dataset_name ImageNet --dataset_path path/to/ImageNet --model_ckpt_path model_ckpts/ImageNet_eps3.pt --objective_type logit
```

Our enhancement approach essentially involves maximizing the logit value of the ground truth class. You can also minimize the cross-entropy loss by setting --objective_type cross_entropy. For the fine-grained datasets we used in the learning experiments, we used "--objective_type logit_diverge", such that the logits of competing classes are explicitly minimized - this seems to produce more compelling perturbations in fine-grained tasks. For example, to enhance HAM10000 dermoscopy images:
```
python scripts/enhance.py --eps 8 --num_steps 16 --dest_dir data/enhanced_dermoscopy_images --dirmap_path path/to/dirmap.csv --dataset_name HAM10000 --dataset_path path/to/HAM10000 --model_ckpt_path model_ckpts/HAM10000_eps1.pt --objective_type logit_diverge
```

The additional script imgproc_code/scripts/enhance_vit_aug.py is similar to enhance.py, but it is designed specifically to use a vision transformer model called [XCiT](https://papers.neurips.cc/paper/2021/file/a655fbe4b8d7439994aa37ddad80de56-Paper.pdf) and implements an array of multi-view augmentations to generate gradient steps for higher-quality perturbations (which seems to be necessary specifically for transformer-based models.) Here is an example terminal command to run this script on some ImageNet animal images (if you run out of GPU memory, try reducing the batch size): 
```
python scripts/enhance_vit_aug.py --dest_dir data/imagenet16_xcit_tuned --eps 20 --step_size 0.5 --num_steps 80 --num_augs 10 --batch_size 8 --dirmap_path data/imagenet16/dirmap.csv --objective_type logit --save_originals --dataset_name ImageNet --dataset_path data/imagenet16 --arch xcit_large_12_p16_224 --model_ckpt_path model_ckpts/debenedetti/xcit-l12-ImageNet-eps-4.pth.tar
```

### Method 2: bash scripts for enhancing datasets and uploading to S3 with multiple perturbation sizes

We provide several bash scripts that automate the process of enhancing entire datasets with multiple different perturbation sizes (pixel budget values ϵ), and uploading the resulting copies of the dataset to S3 to be used in psychophysics experiments. They are found in the imgproc_code/scripts/batch_enhance directory. Some modification of the dataset/csv/model checkpoint paths will be required to get these scripts working on your system. 

## How to predict difficulty of images using robust networks

We introduce a simple difficulty prediction metric: the logit value associated with the groundtruth class from a robust ANN. This can be calculated for images from many datasets using imgproc_code/scripts/test_model_on_dirmap_get_gt_logit.py (which also serves as a way to evaluate a trained network on a dataset). 

Example terminal commands (to be run from inside imgproc_code, after obtaining/setting up these datasets and downloading model checkpoints): 

```
# Get logits for ImageNet Animals (16 classes):
python scripts/test_model_on_dirmap_get_gt_logit.py --dirmap_path path/to/dataset_dirmap.csv --dataset_name ImageNet --model_ckpt_path model_ckpts/ImageNet_eps3.pt --class_num_col orig_class_num

# Get logits for "Idaea4" moth photos (4 classes), and also generate a class confusion matrix:
python scripts/test_model_on_dirmap_get_gt_logit.py --dirmap_path data/idaea4/idaea4_natural/dirmap.csv --dataset_name idaea4 --dataset_path path/to/idaea4_natural --model_ckpt_path model_ckpts/idaea4_eps1.pt --confusion_matrix
```
Note that "--class_num_col orig_class_num" is specified for ImageNet so that we evaluate the ground truth logits on the original 1000 classes, not the superclasses (i.e., how confident is the model that a specific image is "Siberian Husky" rather than "dog" in general).

In order to use robust_gt_logit for psychophysics experiments in this codebase, it must first be explicitly converted to a difficulty measure (a new column in the dirmap called "difficulty") that should have higher values for more difficult images, unlike robust_gt_logit which is higher for easier images. You can add the difficulty column (using a simple normalization calculation, see script for details) with the command: 

```
python scripts/calc_difficulty_from_logit.py path/to/your/dataset_dirmap.csv
```


## How to train/fine-tune robust models

This project builds directly on top of the [Robustness library](https://github.com/MadryLab/robustness) by the Madry Lab - if you wish to experiment with adversarially trained models more broadly, you may be better off using that library directly. Our code provides additional functionality for adversarial fine-tuning of pretrained models.

You can fine-tune an existing model using adversarial training via the script **imgproc_code/scripts/robust_transfer_learning.py**. Please review the arguments list of this script to understand how to use it. Note that, before training/fine-tuning a model on a new, outside dataset, you must format the dataset in a specific way (see [Dataset organization](#dataset-organization-and-how-to-add-new-datasets-to-this-project)).

For example, here is how to adversarially-fine-tune an adversarially-ImageNet-pretrained model on the MHIST histology dataset (with an adversarial ϵ=1 during fine-tuning):
```
python scripts/robust_transfer_learning.py --dataset_name MHIST --dataset_path path/to/imagefolder/formatted/mhist --n_epochs 50 --lr 0.001 --custom_lr_multiplier "" --batch_size 16 --eps 1 --saved_model_ckpt_path model_ckpts/ImageNet_eps3.pt
```

You can also adversarially train models from scratch using this same script. For example, here is how to replicate our adversarially training run of a ResNet50 model on the iNaturalist dataset from scratch (run from inside imgproc_code directory):
```
# In one shot (may take a few weeks)
python scripts/robust_transfer_learning.py --eps 1 --attack_steps 7 --attack_lr 0.3 --n_epochs 200 --lr 0.1 --step_lr 50 --step_lr_gamma 0.1 --gpu_ids 0 --custom_lr_multiplier "" --batch_size 256 --val_batch_size 128 --n_workers 16 --dataset_name inat --dataset_path path/to/inat2021

# ALTERNATIVELY, in two subsequent jobs (For the second job, change "85ad31ec-6919-52878a26a9f8" to the output directory of the first job)
python scripts/robust_transfer_learning.py --eps 1 --attack_steps 7 --attack_lr 0.3 --n_epochs 100 --lr 0.1 --step_lr 50 --step_lr_gamma 0.1 --gpu_ids 0 --custom_lr_multiplier "" --batch_size 256 --val_batch_size 128 --n_workers 16 --dataset_name inat --dataset_path path/to/inat2021

python scripts/robust_transfer_learning.py --eps 1 --attack_steps 7 --attack_lr 0.3 --n_epochs 100 --lr 0.001 --step_lr 50 --step_lr_gamma 0.1 --gpu_ids 0 --custom_lr_multiplier "" --batch_size 256 --val_batch_size 128 --n_workers 16 --dataset_name inat --dataset_path path/to/inat2021 --continue_same_dataset --saved_model_ckpt_path train_output/85ad31ec-6919-52878a26a9f8/checkpoint.pt.latest
```
In this example, we train with an adversarial ϵ=1 (7 attack steps with step size 0.3), for 200 epochs, with a starting learning rate (--lr) of 0.1. We multiply the learning rate by a factor of 0.1 (--step_lr_gamma) every 50 epochs (--step_lr). 


## Dataset organization, and how to add new datasets to this project

In order to add new image classification datasets to this project, there are three requirements: 
1. Datasets must be indexed using "dirmap" csv files (see below)
2. Datasets must be arranged in "[ImageFolder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html) format" (root dataset folder contains split folders like "train", "val", "test", within each of these is one folder per class, containing all images of that class within that split)
3. In most cases, a new class for each dataset must be added to imgproc_code/robustness/robustness/datasets.py 

If you only want to run psychophysics experiments without using the code in imgproc_code, requirements 2 and 3 do not matter. 

### "Dirmap" indexing for datasets

In this codebase, datasets are generally organized using "dirmap" csv files for each dataset (imported as Pandas dataframes). A dirmap contains one row per image, with several important columns that define image attributes:
* split &emsp; (set to "train", "val", or "test")
* class &emsp; (string name of the image's class)
* class_num &emsp; (integer ID associated with the image's class)
* im_path &emsp; (path to the image file, relative to the directory the dirmap csv file is placed in. e.g. "val/dog/ILSVRC2012_val_00000269.JPEG")

Optional columns, depending on the dirmap's origin and what it is being used for, include: 
* orig_class &emsp; (string name of the original class before reassignment - e.g., in ImageNet Animals, class=dog when orig_class=Siberian_husky)
* orig_class_num &emsp; (integer ID associated with the original class)
* orig_im_path &emsp; (relative path to the image file before dataset reorganization)
* url &emsp; (url to access the image online, often from an S3 bucket. A script for uploading image datasets to S3, while adding this column to the dirmap, can be found at imgproc_code/scripts/upload_images_s3.py)
* robust_gt_logit &emsp; (Ground truth logit from a robust model, used to calculate difficulty. Can be added using imgproc_code/scripts/test_model_on_dirmap_get_gt_logit.py)
* difficulty &emsp; (A difficulty score, which in our case would be a normalized version of robust_gt_logit)
* Additional dataset-specific columns (e.g., the MHIST dataset's dirmap has a column "Number of Annotators who Selected SSA (Out of 7)" indicating expert agreement on the image labels).

Dirmap files are initially produced using scripts placed in imgproc_code/dataset_setup, such as define_imagenet_subset_dataset.py (for ImageNet Animals, use 'i16' option), define_ham10000_dataset.py (for HAM10000 dermoscopy dataset), etc. Some of the scripts will work for multiple datasets - for example, imgproc_code/dataset_setup/imagefolder_style_df.py will produce a valid dirmap for any dataset that is already in "[ImageFolder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html) format."

In order to train/evaluate models on image datasets within the [Robustness library](https://github.com/MadryLab/robustness) (which this repository builds upon, and contains a modified version of), datasets must be organized in "[ImageFolder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html) format" (root dataset folder contains split folders like "train", "val", "test", within each of which is one folder per class, each containing all images of that class within that split). The script **imgproc_code/dataset_setup/build_dataset.py** is designed to create ImageFolder-formatted copies of datasets that do not start out with this format, using an initial "recipe" dirmap csv produced by other scripts in imgproc_code/dataset_setup. It must be given the path to the recipe dirmap csv, as well as the path to the dataset root. It also has other features, such as the ability to sample a class-balanced subset of the dataset, resize/reformat images, and select a subset of classes to include - see arguments list in the script for details. 

### Setting up new datasets for model training/evaluation

To train robust models on new, outside datasets, you must define a new class in imgproc_code/robustness/robustness/datasets.py. You can also add hyperparameter defaults in imgproc_code/robustness/robustness/defaults.py, and custom data augmentations in imgproc_code/robustness/robustness/data_augmentation.py. See also the original [Robustness library documentation](https://robustness.readthedocs.io/en/latest/example_usage/training_lib_part_2.html#training-on-custom-datasets).







# Running psychophysics experiments (see "psych_code" directory)

The psych_code directory contains code to replicate our psychophysics experiments.
The codebase is designed to streamline the process of setting up and running new experiments using Amazon Web Services (AWS).
In addition to HTML/Javascript code that enables the experiments to run in a web browser, we include scripts for automatically setting up the infrastructure to host these experiments online through the AWS platform, such that participants can then be recruited from Prolific or Mechanical Turk. It is also convenient to use the same setup to run the experiment locally for in-person participants. 

## Quick start: hosting pre-designed experiments from the L-WISE paper on Prolific and AWS

1. Make sure you have [set up your AWS credentials](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html) such that you can call AWS from the command line. 
2. Create and configure an experiment on the Prolific website, and obtain the completion code that participants who complete the experiment should receive. 
3. Run the psych_code/scripts/deploy_experiment.py script from within the psych_code directory. Here is an example command that deploys the moth classification learning experiment from the L-WISE paper to AWS:
```
python scripts/deploy_experiment.py --experiment_name idaea4_learn --experiment_number 10 --num_trialsets 280 --delete_old_apis --num_conditions 7 --randomization_block_size 7 --alternate_within_blocks --completion_code <COMPLETION_CODE_HERE>
```
4. After the script finishes, test the experiment interface in your browser using the provided testing link. 
5. Use the psych_code/scripts/get_experiment_data.py script to try downloading data from your testing session. For example:
```
python scripts/get_experiment_data.py --experiment_name idaea4_learn --experiment_number 10
```
6. Use code similar to that found in notebooks/make_figs.ipynb to convert the data to a Pandas dataframe, and make sure it looks as you would expect. 
7. Copy the Prolific URL (provided by the deploy_experiment.py script) into the Prolific website. You can find the logged output of the deployment script in the corresponding directory in psych_code/deployed_experiments (look for a "deployment.log" file). If everything went well, you are ready to recruit participants!
8. Run get_experiment_data.py again to download data once a number of participants have completed the task. It is advisable to do this after a small number of participants have finished to make sure everything is working correctly. 

**IMPORTANT NOTE**: By default, the experiments are set up to log data after each individual trial. This aims to conserve as much data as possible, but can also make the experiment run slowly (especially if there are lots of trials). To speed up the experiment, consider omitting the "&trialsubmit=(url here)" part of the experiment url - it will then only save all the data at the end of the session (the disadvantage of this is that, if a participant does not make it to the end of the session for any reason, you will lose their data).

### How to deploy experiments to reproduce L-WISE paper results: 

Deploying a different experiment using the steps above requires only varying the arguments passed to deploy_experiment.py. Below are example commands for deploying each of the experiments from the L-WISE paper. Note that --aws_prefix can be set to anything (e.g., your name) - it is just a way of keeping track of which resources are yours on a shared AWS account. However, if you change it from the default "lwise", you must also change the aws_prefix value near the top of the html file for the experiment. 

**ImageNet animal recognition task (Robust-ResNet-50 image enhancement with different ϵ pixel budgets)**
```
python scripts/deploy_experiment.py --experiment_name imagenet_animals_main --experiment_number 10 --aws_prefix lwise --num_trialsets 100 --delete_old_apis --num_conditions 1 --completion_code <<<optional: Prolific completion code here>>> --screen_out_code <<<optional: Prolific screenout code here>>>
```

**Moth species learning task**
```
python scripts/deploy_experiment.py --experiment_name idaea4_learn --experiment_number 10 --aws_prefix lwise --num_trialsets 280 --delete_old_apis --num_conditions 7 --randomization_block_size 7 --alternate_within_blocks --completion_code <<<optional: Prolific completion code here>>>
```

**Dermoscopy learning task**
```
python scripts/deploy_experiment.py --experiment_name ham4_learn --experiment_number 10 --aws_prefix lwise --num_trialsets 280 --delete_old_apis --num_conditions 7 --randomization_block_size 7 --alternate_within_blocks --completion_code <<<optional: Prolific completion code here>>>
```

**Histology learning task**
```
python scripts/deploy_experiment.py --experiment_name mhist_learn --experiment_number 10 --aws_prefix lwise --num_trialsets 120 --delete_old_apis --num_conditions 2 --randomization_block_size 2 --alternate_within_blocks --completion_code <<<optional: Prolific completion code here>>>
```

**Screening task (binary turtle species classification)**
```
python scripts/deploy_experiment.py --experiment_name turtlequal --experiment_number 10 --aws_prefix lwise --num_trialsets 1 --delete_old_apis --num_conditions 1 --completion_code <<<optional: Prolific completion code here>>>
```

### Tools for hosting experiments on Mechanical Turk

Prolific is the default platform for this codebase - we only used MTurk to run pilot experiments. If you wish to run experiments on MTurk, there are several scripts in psych_code/scripts/mturk that may be useful:
* create_hit.py creates a new HIT using a URL that you can obtain by running deploy_experiment.py (or see deployment.log afterwards). You must specify the path of config.yaml manually: it will pull various details of the MTurk HIT from this file, which you can edit as desired. PLEASE NOTE: by default, create_hit.py deploys to the mturk SANDBOX, not the production environment. Use the --prod flag when you are ready to switch. 
* hit_status.py allows you to monitor an ongoing HIT.
* get_hit_data.py retrieves all data from a specific HIT, directly from the MTurk platform's storage. Note that this is a separate, redundant data storage mechanism: you can get the same data from S3 using psych_code/scripts/get_experiment_data.py. More usefully, get_hit_data.py also can be called with --approve_all_assignments (to approve and pay each participant), --pay_bonuses (pay bonus to each participant), and/or assign_qualification (which assigns a qual to all participants in the HIT: create a new one by specifying --qualification_name, or use an existing one by specifying --qualification_type_id).
* compensation_hit.py allows you to create and manage "compensation HITs", which are special HITs created specifically to compensate a participant who spent time on the experiment but wasn't able to finish for whatever reason (e.g., if the experiment froze or crashed). 
* mturk_qual.py allows you to manage qualifications: you can create new quals and assign them to mturk workers manually using this script. 


## Obtaining and analyzing public data from the original L-WISE experiments

We provide a publicly-available, de-identified copy of all of the data we collected from human participants for the L-WISE paper. You can run the commands below to download and extract this dataset (make sure your working directory is at the repository root). You can then run notebooks/make_figs.ipynb, which performs all of the data analyses from the paper and generates the associated figures. 

```
mkdir -p psych_data
wget https://github.com/MorganBDT/L-WISE/releases/download/v1.0.1/L-WISE_deidentified_psych_data.zip
unzip L-WISE_deidentified_psych_data.zip -d psych_data
rm L-WISE_deidentified_psych_data.zip
```

The data from each of the behavioral experiments is stored as a .csv file loosely in psych_data. "i16" is 16-way ImageNet animal classification, "idaea4" is moth classification, "ham4" is dermoscopy, and "mhist" is histology. Each row in these dataframes corresponds to one trial completed by one participant. The most important columns are: 
* participant: an integer indicating which participant this trial is from.
* obs: integer trial number (within each participant).
* block: which block the trial is part of, corresponding to different blocks in session_config within the config.yaml file for the experiment.
* condition_idx: which experimental group the participant is in. 0 is always the control group. See main paper and make_figs.ipynb for details of groups in different experiments.
* trial_type: type of trial. This corresponds to trial_type settings in the config.yaml file for the experiment, specified either as trial_types or conditional_trial_types in each block of session_config. 
* class: the class of the stimulus presented to the participant. 
* stimulus_name: the name of the stimulus presented to the participant. This may be an alias rather than the original class name. 
* stimulus_image_url: URL to the stimulus image presented to the participant.
* choice_name: the name of the choice made by the participant (maps to "stimulus_name" which can be an alias, not necessarily to "class"). 
* i_choice: the integer index of the choice made by the participant (1:1 mapping with classes or choice names)
* i_correct_choice: the integer index of the correct choice for that trial
* perf: 1 if the participant's response was correct, 0 if incorrect or no response.
* reaction_time_msec: number of milliseconds before the participant made a response, starting from when the response options became available. 
* rel_timestamp_response: time of response relative to the start of the experiment session

## Subdirectories in experiment_files (one for each experiment)

The experiment_files directory contains one subdirectory for each of several psychophysics experiments. Each directory contains, at minimum: 
* An HTML file with the same name as the subdirectory, containing all of the HTML and Javascript code to run the experiment (aside from plugins, libraries etc)
* A dataset_dirmap.csv file, indexing all of the images available within that experiment. Alternatively, you can include custom trial sequences by including trialsets.csv instead (see psych_code/experiment_files/example_custom_experiment_0 for an example of this). 
* A config.yaml file, specifying various configuration settings for the experiment. 

Each directory corresponding to one experiment must be named "{experiment_name}_{experiment_number}", where experiment_name is a string and experiment_number is an integer. For example, in "dinosaurtask_v2_learn_3", the experiment_name is dinosaurtask_v2_learn and the experiment_number is 3. 
The combined name of the directory is also called the experiment_id in some parts of the code.
The names often correspond to the dataset being used, or some derivative part of a larger dataset. 
For example, ham4_learn refers to the use of 4 classes from the HAM10000 dataset, and idaea4 refers to using 4 classes from the iNaturalist dataset (all of them being moth species in the genus _idaea_).
The experiment numbers are arbitrary: we used them to track various pilot experiments, iterative versions of experimental code, etc. 

### **Descriptions of experiment_files directories:**

**imagenet_animals_main_10**: The main ImageNet animal recognition experiment, testing logit-max enhancement at different ϵ pixel budgets, with off-the-shelf enhancement algorithms as controls

**imagenet_animals_guide_models_10**: Supplementary ImageNet animal recognition experiment (compare enhancements using several different guide models). Before deploying, copy dataset_dirmap.csv from imagenet_animals_main_10. 

**imagenet_animals_loss_ablation_10**: Supplementary ImageNet animal recognition experiment (loss function ablation: test cross-entropy vs logit-max losses for train set and val set images). Before deploying, copy dataset_dirmap.csv from imagenet_animals_main_10. 

**idaea4_learn_10**: iNaturalist moth category learning experiment

**ham4_learn_10**: Main HAM10000 dermoscopy category learning experiment

**ham4_learn_pilot_0**: Pilot HAM10000 dermoscopy image category learning experiment, unsuccessful at boosting learning because initial enhancement level was too strong at ϵ=20 (results presented in supplementary materials of L-WISE paper). Before deploying, copy dataset_dirmap.csv from ham4_learn_10.

**mhist_learn_10**: MHIST Histology category learning experiment

**turtlequal_10**: Qualifier task in which participants were asked to learn the difference between two different species of turtles (loggerhead and leatherback, images from ImageNet validation set)

**example_custom_experiment_0**: Example of an experiment with a fully customized trial sequence. See ["Implementing experiments with customized trial sequences"](#implementing-experiments-with-customized-trial-sequences) below. The same html file can also be equivalently used with dataset_dirmap.csv as with the above experiments. 

## HTML files for experimental code

Each experiment has its own HTML file which contains all of code for running the experimental interface in a browser. These were derived from the HTML files in the [Wormholes](https://github.com/ggaziv/Wormholes/tree/main/psych) project and modified extensively. They are designed to be hosted online (e.g., using AWS S3), and to interact with other AWS services to receive session parameters (e.g., random condition assignments for each participant) and store behavioral data. If you wish to develop your own experiment, we recommend starting with psych_code/experiment_files/example_custom_experiment_0/example_custom_experiment_0.html (this is the most flexible and up-to-date version - it works with mouse responses or f/j keyboard responses, and with either dataset_dirmap.csv or trialsets.csv).

## Dataset_dirmap.csv and trialsets.csv

Each of the directories in psych_code/experiment_files that correspond to experiments from the L-WISE paper typically contains a file called dataset_dirmap.csv. This file contains a record of all images that can be seen by participants in the corresponding experiment - when the actual "trialsets" (sequences of trials seen by each individual participant) are generated, it is done by randomly sampling rows from this file. 

A more flexible alternative is to specify your own custom trial sequence - in which case you would include a "trialsets.csv" file in your experiment's directory in experiment_files instead of dataset_dirmap.csv. Please see ["Implementing experiments with customized trial sequences"](#implementing-experiments-with-customized-trial-sequences) below for more details on this approach. 

## Configuring the experiments using config.yaml

The directory for each experiment must contain a config.yaml file, which contains many configurable parameters for the experiment such as stimulus timings and trial block structure. 

### Global and trial-block-specific settings

config.yaml has two main sections: session_config and trial_config. trial_config contains global settings that affect every trial in the experiment. Session_config contains settings that apply only to specific blocks of trials, each of which is a sub-section in of session_config in config.yaml, with an arbitrary name (e.g. train_1, test_3, warmup_block, etc). Most of the settings in trial_config can also appear in the blocks within session_config: generally speaking, **values in trial_config serve as defaults that can be overridden on a block-by-block basis within session_config.** These settings can then be further overridden for specific types of trials - see ["Specifying different trial types"](#specifying-different-trial-types) below. 

Each block in the experimental task consists of some number of trials, which may be drawn from one or more "splits" (i.e., train, val, test). Note that this generally refers to the train/test splits of image datasets designed for machine learning, rather than "training" or "testing" human participants. Currently, there is only support for class-balanced experiments where each block contains equal numbers of images from each class (if this doesn't work for you, see ["Implementing experiments with customized trial sequences"](#implementing-experiments-with-customized-trial-sequences) below). You can also specify some number of "calibration trials", which are currently always "circle vs triangle" (note that n_calibration_trials must be a multiple of 2). As an additional control, you can specify n_repeats_of_one_stim. If this is greater than 0, one image will be randomly selected and held out from the experiment: this image will be presented repeatedly within the block. If n_repeats_of_one_stim is greater than 0 in subsequent blocks, it will still be the same stimulus that was presented repeatedly in previous blocks. All trial types are shuffled within their block, regardless of class, split, trial type, and whether they are normal/calibration/repeat trials.

For example, let's say our task has 3 classes: dog, wolf, and coyote. The first block in session_config has n_trials_per_class_train=2, n_trials_per_class_val=0, n_trials_per_class_test=1, n_calibration_trials=2, and n_repeats_of_one_stim=3. The resulting block will have a total of 14 trials: 2 trials with training set images from each of the 3 classes (6 trials), plus 1 trial with a test set image from each class (3 trials), plus 2 calibration trials, plus 3 repeat stimulus trials. All of the 14 trials will be shuffled into a random sequence. 

In addition to session_config and trial_config in config.yaml, there is also a hit_config section at the bottom, which can be used to set various parameters of an experimental "HIT" specifically on Mechanical Turk (not relevant for Prolific/other deployments).

If you specify your trial sequences using trialsets.csv (see ["Implementing experiments with customized trial sequences"](#implementing-experiments-with-customized-trial-sequences) below), you do not need to include the session_config section of config.yaml, only trial_config. 

### Class/choice names and aliases

This codebase is designed for experiments in which the participant must choose among two or more classes on each trial. There are several settings in config.yaml that determine how these choices are displayed. choice_names_order is an ordered list of class names (the names are as they appear in the "class" column in dataset_dirmap.csv, but with underscores replaced with spaces): the buttons the participant must click on to make a response appear in this order arranged in a ring around the center of the screen. If shuffle_choice_order is set to true, their positions are randomly shuffled between each trial. If rotate_choice_order is true, the ring is rotated randomly such that the button positions are "randomized" but their relative ordering stays the same (as specificed in choice_names_order). 

Optionally, each class can be assigned an alias (alternative name for the class). These aliases are defined in the choice_names_aliases dictionary in the trial_config section of config.yaml (see the idaea4 experiment for an example). If choice_aliases_random_shuffle is set to true, the mapping between class names and aliases will be randomly shuffled for each participant (but always consistent within one participant session trialset). 

### Keyboard control for binary classification tasks

For binary classification tasks, it is probably faster for most participants to press the "F" or "J" key to enter their response to each trial rather than using the mouse. Assuming you build your experiment starting with experiment_files/example_custom_experiment_0/example_custom_experiment_0.html, you can set keypress_fj_response to "true" if you want to use F/J (or false if you want to use the mouse). 

### Specifying different trial types

Many experiments require multiple distinct "types" of trials - for example, one type of trial may involve presenting an enhanced image while another involves presenting a non-enhanced image. To specify multiple trial types for a given block in session_config, you can add a sub-section within the block section called "trial_types", with a sub-sub-section for each trial type, containing any settings that should override the block-level (session_config) or global (trial_config) settings for that specific trial type. One important thing to set might be the "bucket" variable: this is how we controlled modifiable parameters of the images being presented (e.g., the pixel budget by which the image was enhanced, the guide model or loss function that was used to enhance it, etc). Essentially, we set up buckets that contain differently-modified copies of the same dataset with exactly the same file names and directory structure - e.g., one bucket for original images, one for images enhanced by ResNet-50 with ϵ=10, another for ϵ=20, etc - each containing a set of image files with exactly the same names/paths. When "bucket" is set at the trial-type level, after the image is selected from dataset_dirmap.csv (which contains a "url" column with a url to the corresponding image), the S3 bucket name will be replaced within the URL with whatever "bucket" is set to. The config.yaml files for the "imagenet_animals" experiments have good examples of this setup. 

For other experiments (e.g., the "ham4", "idaea4", and "mhist" learning experiments), settings such as the S3 bucket to use or other variables should depend on which experimental group the participant is in (e.g., control group, L-WISE group, etc). For these experiments, each participant is given a "condition_idx" integer value denoting which group they are in, and this value is used to select trial types from a block-level sub-section called "conditional_trial_types." See the config.yaml files of the idaea4 moth experiments for an example. Although in this case only one possible trial type for each block is specified for each experimental group, it is possible to add more entries to each dictionary within the conditional_trial_types dict, allowing participants to be shown a mixture of different trial types - this is similar to the case where we specify trial_types in the imagenet animal experiments (but the composition of which now depends on the experimental group they are in). 

If there are multiple trial types in a block, they will be balanced within each class and each split (considering regular trials only, not calibration or repeat stimulus trials). Therefore, for each train/val/test split, the number of trials per class must be set to a multiple of the number of trial types. If this does not work for you, you can specify a custom trial sequence (see ["Implementing experiments with customized trial sequences"](#implementing-experiments-with-customized-trial-sequences) below). 

## Deploying experiments to online platforms

### Initial deployment to AWS

The script psych_code/scripts/deploy_experiment.py is designed to automate the process of deploying an experiment online through Amazon Web Services (AWS). **The script automates the following main steps**:
1. Create a new S3 bucket to host the experiment, and upload the main .html file, config.yaml, and dataset_dirmap.csv to the new bucket. 
2. Generate a "trialset" for each participant slot (specified by an integer "trialset_id"). These are .csv files (also stored in .js format for import in the main .html) with one entry for each trial, containing all of the details about that trial including the url of the image to be displayed and many other parameters that are specified in config.yaml. Notably, it is advisable to create way more trialsets (perhaps three or four times as many) than the number of participants who are expected to be recruited: some participants may leave without completing the task, their browser might have some kind of error, etc. Currently there is no automated way to add more trialsets other than re-deploying the whole experiment, and if the trialset counter (see below) counts above the number of trialsets available, this will cause an error if more participants try to join the experiment. 
3. Upload all trialsets to the new S3 bucket. 
4. Create a new "trialset_counter" DynamoDB table. This table has only one entry: the current trialset_id indicating the trialset to be assigned to the next participant. This table's job is to ensure that each trialset is assigned to one participant and one participant only, even if two participants start at exactly the same time. 
5. Create a new "trialset_id_mapper" DynamoDB table. This table stores a record of each participant's completion of the experiment, including a unique ID for that session, the participant ID (e.g., a worker ID on Mechanical Turk), trialset_id they were assigned, which experimental group they are in, links to the session's data that will be uploaded to S3 upon completion, and so on. 
6. Create a new AWS Lambda function (code in psych_code/modules/session_metadata_lambda.py) which will be called in order to assign the trialset_id at the beginning of the session and save the data after every trial (and at the end upon completion). In the process of deploying a lambda function, a new IAM role and API Gateway API will also be created to enable the Lambda function to be called using a specific URL. 
7. Create and upload an aws_constants.js file, containing the url at which to call the newly created Lambda function. 

The script will return a URL at which the experiment has been deployed - putting this URL into your browser should enable you to try the experiment. 

Running deploy_experiment.py requires that you first [set up AWS credentials](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html) on your machine such that you can perform actions on an AWS account, including creating new S3 buckets, Lambda functions, DynamoDB tables, IAM roles, and API Gateway APIs. Notably, a copy of everything that is going to be uploaded (trialsets, html, dataset_dirmap.csv, config.yaml, aws_constants.js) is also saved locally to the psych_code/deployed_experiments directory as an archival record of what was deployed. **This local copy will, however, be overwritten upon a subsequent deployment of the same experiment_id unless it is renamed or moved elsewhere.**

Here is an example command to deploy an experiment with experiment_name=idaea4_learn and experiment_id=10:

```
python scripts/deploy_experiment.py --experiment_name idaea4_learn --experiment_number 10 --aws_prefix lwise --num_trialsets 406 --delete_old_apis --num_conditions 7 --randomization_block_size 7 --alternate_within_blocks --completion_code <COMPLETION_CODE_HERE>
```

As you can see, deploy_experiment.py allows you to specify the number of trialsets to be created, which must be a multiple of --num_conditions (the number of distinct experimental groups participants can be assigned to). By default, there are an equal number of trialsets created for each experimental group, and they are randomly ordered (in terms of the integer trialset_ids). This can lead to imbalances in the number of participants assigned to each group. This problem can be mitigated by setting --randomization_block_size to some multiple of --num_conditions (which must also be a factor of --num_trialsets): the order of experimental groups will be shuffled randomly within "blocks" of this number of trialsets. For example, let's say there are three conditions A, B, and C. We set --num_trialsets to 9. If we are very unlucky, we might randomly generate trialsets such that the sequence of group assignments is AAABBBCCC (the first three participants in the experiment would all be assigned to group A). Remember we want to generate way more trialsets than the number of expected participants - if we only ended up recruiting 6 participants and none of them left or encountered a bug, we wouldn't get any participants in group C. If we use --randomization_block_size 3, we might get a sequence like BCA|ABC|CAB (where | delineates randomization block boundaries). --alternate_within_blocks imposes even more structure such that the trialsets alternate within each block: in our example, we would be guaranteed to get the order ABC|ABC|ABC if we use this flag. While useful for ensuring a good balance of participants assigned to each group, this predictable order could introduce selection biases if the order of participant recruitment is not random (e.g., in-person recruitment), but is very unlikely to be a problem for massively parallel online recruitment on MTurk or Prolific because the precise order in which participants join the experiment can be assumed to be random. 

The --aws_prefix appends a prefix to the names of all created AWS resources (Lambda functions, S3 buckets, etc) - this is useful on AWS accounts shared by multiple people to easily keep track of who created what. You must also change the aws_prefix in your html file to whatever you set this to. --delete_old_apis ensures that any old APIs left over from previous deployments with the same experiment_id are cleaned up (otherwise this takes forever manually). The --completion_code is used for experiments to be deployed on Prolific (this is the code participant's receive when they complete the experiment session - it is provided when you set up the experiment on the Prolific website).  

Note also that you can use the --local_only flag to only set up trialset_ids, aws_constants.js etc locally (i.e., everything will appear in psych_code/deployed_experiments): this is useful for testing the trial generation configuration. If you add --aws_config_only, the script does nothing but update the AWS configuration (Lambda function, DynamoDB, etc) - it does not generate new trialsets, or change any of the contents of the experiment's S3 bucket. Finally, you can specify --files_only: this only generates the trialsets and uploads them to S3 along with the html file, config.yaml, etc (this is much faster if you have already set up all the AWS resources). 

Here is another example of an experiment deployment command: 

```
python scripts/deploy_experiment.py --experiment_name imagenet_animals_main --experiment_number 10 --aws_prefix lwise --num_trialsets 100 --delete_old_apis --num_conditions 1 --completion_code <COMPLETION_CODE_HERE> --screen_out_code <SCREENOUT_CODE_HERE>
```

This deploys an ImageNet Animal experiment in which there is only one experimental group (each participant is shown both original and modified images interspersed with each other). This experiment also has a screening phase: we provide two different codes from the Prolific website, allowing us to later compensate participants differently depending on whether they completed the entire experiment or were screened out. 

The psych_code/scripts/deployed_experiment_teardown.py script is provided as a way of efficiently deleting all AWS resources associated with a given experiment. It is to be used with caution because it will **permanently** delete any and all participant data that was collected and stored in that experiment's S3 bucket. 

### Downloading data from deployed experiments

To download participant data for an experiment, use psych_code/scripts/get_experiment_data.py. By default, this will fetch data only from completed experiment sessions. If --get_partial_trials is used, for participants where a complete session data record is not available, the script will check for partial data saved from individual trials and attempt to reconstruct an equivalent session record (you can use --require_min_test_trials to require a certain number of test trials to be present in order for this reconstruction to be considered valid on a per-participant basis). If --get_all_partial_trials is used, the script runs through this reconstruction procedure for every single participant (very time consuming and redundant in most cases - this option is included primarily for debugging purposes). Data will be downloaded in .h5 format: please see notebooks/make_figs.ipynb for an example of how to convert this into a Pandas dataframe. 


## Key steps when implementing a new experiment

1. Make a new directory in experiment_files, named {experiment_name}_{experiment_number} (you can make the experiment_name any string and experiment_number any integer). You should almost certainly do this by copying an existing directory. 
2. Change the name of the html file in the new directory to {experiment_name}_{experiment_number}.html. Also edit the experiment_name and experiment_number variable declarations in the first "body" block within the html file itself to match. 
3. If you want to use a different dataset or a different set of images for the new experiment, edit or replace dataset_dirmap.csv (see [Dataset organization](#dataset-organization-and-how-to-add-new-datasets-to-this-project) section for information on how this should be structured). Note that this may not always be a drop-in replacement: for example, config.yaml might be set up to use class names, split names, AWS bucket names, etc from the original dataset, **you will probably need to create new graphics for the class choice buttons**, etc. Code in the scripts/icon_generation folder might be useful for making the new graphics. A further option is to manually specify the trial sequence for each participant session (see ["Implementing experiments with customized trial sequences"](#implementing-experiments-with-customized-trial-sequences) below). 
4. Edit config.yaml as desired. See ["Configuring the experiments using config.yaml"](#configuring-the-experiments-using-configyaml) above for more information. 

For more extensive modifications to the interface/behavior of the experimental task, we suggest [using trialsets.csv to specify your own trial sequence](#implementing-experiments-with-customized-trial-sequences) and/or modifying the .html files. 

## Implementing experiments with customized trial sequences

Most of the instructions above apply to experiments with a specific structure that has several constraints (e.g., balanced classes and trial types within each block, dataset must be specified in a dataset_dirmap.csv with different versions of the same images stored with the same name in different S3 buckets, etc..).

A more flexible way to implement your own experiment is to directly specify the sequence of stimuli to be shown, trial by trial, participant by participant, in a file called "trialsets.csv" inside the subdirectory of experiment_files corresponding to your experiment. For an example of this, please see psych_code/experiment_files/example_custom_experiment_0. Here, trialsets.csv specifies two trialsets (sessions for two different participants): the first session has 16 trials, and the second has 32 (not that you'd necessarily want participants to see different numbers of trials - this is just to show that you can put in whatever trials you want). Each session (or "trialset") is specified by a unique integer value in the trialset_id column. You must also specify "condition_idx", "block", "class", and "url" for each row (each row in trialsets.csv corresponds to one trial). "condition_idx" can be set to 0 if you aren't assigning participants in multiple condition groups, and "block" can be set to 0 if you don't have distinct blocks of trials within your experiment sessions. You can also optionally specify "trial_type" for each row (e.g. "natural_image", "enhanced_image", "calibration_trial", or any arbitrary string).

**VERY IMPORTANT: You can add more columns to trialsets.csv as desired - these can be given the same names as any of the properties in the trial_config block inside the config.yaml file (the values in trialsets.csv will overwrite any defaults in the config.yaml file on a trial-by-trial basis).** 

Once you have set up your trialsets.csv, you can deploy the experiment as usual, but with the additional "--prespecified_trialsets" flag. Below are a few commands to deploy example experiments specified in this way.

1. Binary turtle classification experiment (we show the fixation cross on every second trial, as a demonstration of how to override variables in the trial_config block in config.yaml: see experiment_files/example_custom_experiment_0/trialsets.csv):
```
python scripts/deploy_experiment.py --experiment_name example_custom_experiment --experiment_number 0 --aws_prefix lwise --prespecified_trialsets
```

2. Binary turtle classification experiment (same as above, but in this version the participant presses F or J keys to make a response in each trial instead of clicking a button with the mouse):
```
python scripts/deploy_experiment.py --experiment_name example_custom_experiment --experiment_number 0 --aws_prefix lwise --prespecified_trialsets --config_file_name config_FJ_keypress_response.yaml
```

3. 16-way ImageNet animal classification experiment (note that we specify  "--data_spec_file imagenet_trialsets.csv" to override the default "trialsets.csv"):
```
python scripts/deploy_experiment.py --experiment_name example_custom_experiment --experiment_number 0 --aws_prefix lwise --prespecified_trialsets --data_spec_file imagenet_trialsets.csv --config_file_name imagenet_config.yaml
```

## Acknowledgements

This work was supported in part by Harvard Medical School under the Dean’s Innovation Award for the Use of Artificial Intelligence, in part by Massachusetts Institute of Technology through the David and Beatrice Yamron Fellowship, in part by the National Institute of General Medical Sciences under Award T32GM144273, in part by the National Institutes of Health under Grant R01EY026025, and in part by the National Science Foundation under Grant CCF-1231216. The content is solely the responsibility of the authors and does not necessarily represent the official views of any of the above organizations. The authors would like to thank Yousif Kashef Al-Ghetaa, Andrei Barbu, Pavlo Bulanchuk, Roy Ganz, Katherine Harvey, Michael J. Lee, Richard N. Mitchell, and Luke Rosedahl for sharing their helpful insights into our work at various times. This codebase borrows extensively from the [Robustness library](https://github.com/MadryLab/robustness) by the Madry Lab, and uses psychophysics code modified from the [Wormholes](https://github.com/ggaziv/Wormholes) project in the DiCarlo Lab. We also include a copy of the [Multiscale Retinex](https://www.ipol.im/pub/art/2014/107/) source code, with minor compatibility modifications. 

## Citation

If you find this code to be useful, please consider giving it a star ⭐️ and a citation as follows: 

_Talbot, Morgan B., Gabriel Kreiman, James J. DiCarlo, and Guy Gaziv. "L-WISE: Boosting Human Visual Category Learning Through Model-Based Image Selection And Enhancement." International Conference on Learning Representations (2025)._

BibTeX:
```
@inproceedings{talbot2025wise,
  title={L-WISE: Boosting Human Visual Category Learning Through Model-Based Image Selection And Enhancement},
  author={Talbot, Morgan B and Kreiman, Gabriel and DiCarlo, James J and Gaziv, Guy},
  booktitle={International Conference on Learning Representations},
  year={2025}
}
```

