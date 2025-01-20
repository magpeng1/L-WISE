#!/bin/bash

# Default values
DATASET_NAME="default_dataset"
CLASS_INDICES=()
GPU_ID=0
UPLOAD_ENHANCED=0

# Display usage information
usage() {
  echo "Usage: $0 -n dataset_name -c class_index [-c class_index ...] [--gpu_id GPU_ID] [--upload_enhanced]"
  echo "Example: $0 -n leopardjaguar -c 288 -c 290 --gpu_id 1 --upload_enhanced"
}

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    -n) DATASET_NAME="$2"; shift ;;
    -c) CLASS_INDICES+=("$2"); shift ;;
    --gpu_id) GPU_ID="$2"; shift ;;
    --upload_enhanced) UPLOAD_ENHANCED=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown parameter passed: $1"; usage; exit 1 ;;
  esac
  shift
done

# Ensure CLASS_INDICES is not empty
if [ -z "${CLASS_INDICES[*]}" ]; then
  echo "Class indices must be specified with -c option."
  exit 1
fi

# Adjust DATASET_NAME for bucket naming conventions
BUCKET_NAME_BASE=${DATASET_NAME//_/-}

# Set CUDA_VISIBLE_DEVICES for GPU if not 0
if [ "$GPU_ID" -ne 0 ]; then
  export CUDA_VISIBLE_DEVICES=$GPU_ID
else
  unset CUDA_VISIBLE_DEVICES
fi

# Running commands in sequence with the specified arguments
echo "Defining ImageNet subset dataset by class list..."
python dataset_setup/define_imagenet_subset_dataset_by_class_list.py --dataset_dest_location data/$DATASET_NAME --class_indices "${CLASS_INDICES[@]}"

echo "Building dataset..."
python dataset_setup/build_dataset.py --data_path data/$DATASET_NAME --subset --n_per_class_train 0 --n_per_class_val 50 --n_per_class_test 0

if [ "$UPLOAD_ENHANCED" -eq 1 ]; then
  if [ "$GPU_ID" -ne 0 ]; then
    echo "Enhancing dataset on GPU $GPU_ID..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python enhance.py --dest_dir data --dirmap_path data/$DATASET_NAME/dirmap.csv --dataset_name ImageNet --objective_type logit_diverge --save_originals
  else
    echo "Enhancing dataset..."
    python enhance.py --dest_dir data --dirmap_path data/$DATASET_NAME/dirmap.csv --dataset_name ImageNet --objective_type logit_diverge --save_originals
  fi
else
  echo "Resizing natural images..."
  python resize.py --dest_dir data --dirmap_path data/$DATASET_NAME/dirmap.csv
fi

echo "Uploading natural images to S3..."
python dataset_setup/upload_images_s3.py --data_path data/${DATASET_NAME}_natural --create_new_bucket --bucket_name morgan-${BUCKET_NAME_BASE}-natural

if [ "$UPLOAD_ENHANCED" -eq 1 ]; then
  echo "Uploading enhanced images to S3..."
  python dataset_setup/upload_images_s3.py --data_path data/${DATASET_NAME}_10_0.5_20_logit_diverge --create_new_bucket --bucket_name morgan-${BUCKET_NAME_BASE}-10-0dot5-20-logit-diverge
fi

echo "All tasks completed."
