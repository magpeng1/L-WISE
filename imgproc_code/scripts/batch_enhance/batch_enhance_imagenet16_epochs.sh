#!/bin/bash

# VALIDATION SET Dataset and model settings
DATASET_NAME_RB="ImageNet"  # This is the name of the dataset as seen in the robustness library (e.g., ImageNet)
DATASET_NAME="imagenet16"  # This is the name of the dataset as we will actually name it (e.g., imagenet16)
DATASET_PATH="/home/morgan/projects/learn-histopath-backend/data/imagenet16"
DIRMAP_CSV_PATH="/home/morgan/projects/learn-histopath-backend/data/imagenet16/dirmap.csv"
CHECKPOINT_DIR="/media/KLAB39/morgan_data/learn-histopath-backend/train_outputs/imagenet_resnet50_90epochs_eps3/"

# Control variables
GPU_ID=0
SAVE_ORIGINALS=false  # Set to true to include --save_originals option

# Set save_orig based on SAVE_ORIGINALS
if $SAVE_ORIGINALS; then
    save_orig="--save_originals"
else
    save_orig=""
fi

# Fixed epsilon and number of steps for all checkpoints
eps=20
num_steps=$(printf "%.0f" $(echo "2 * $eps" | bc))

# Checkpoint IDs to use
checkpoint_ids=(0 9 33 89)

for checkpoint_id in "${checkpoint_ids[@]}"; do
    CHECKPOINT_NAME="${checkpoint_id}_checkpoint.pt"
    MODEL_PATH="${CHECKPOINT_DIR}${CHECKPOINT_NAME}"
    
    echo "Running enhancement with checkpoint ${CHECKPOINT_NAME}"
    
    CUDA_VISIBLE_DEVICES=${GPU_ID}, python scripts/enhance.py \
        --eps $eps \
        --num_steps $num_steps \
        --dest_dir "data/${DATASET_NAME}_epochs/${checkpoint_id}_checkpoint" \
        --dirmap_path ${DIRMAP_CSV_PATH} \
        --dataset_name ${DATASET_NAME_RB} \
        --dataset_path ${DATASET_PATH} \
        --model_ckpt_path ${MODEL_PATH} \
        --objective_type logit ${save_orig}
    
    # Only set save_orig for the first run if SAVE_ORIGINALS is true
    save_orig=""
done

if $SAVE_ORIGINALS; then
    echo "Original images are saved in the first checkpoint's output directory"
fi