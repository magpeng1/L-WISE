#!/bin/bash

# Check if required arguments are provided
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 DIRMAP_PATH DATASET_PATH MODEL_CKPT_DIR [GPU_ID]"
    echo "DIRMAP_PATH: Path to the CSV file"
    echo "DATASET_PATH: Path to the directory containing the ImageNet16 dataset"
    echo "MODEL_CKPT_DIR: Directory containing model checkpoints (0_checkpoint.pt, 1_checkpoint.pt, etc.)"
    echo "GPU_ID: Optional GPU ID (default: 0)"
    exit 1
fi

# Assign arguments to variables
DIRMAP_PATH="$1"
DATASET_PATH="$2"
MODEL_CKPT_DIR="$3"
GPU_ID="${4:-0}"  # Set default GPU_ID to 0 if not provided

# Check if the paths exist
if [ ! -f "$DIRMAP_PATH" ]; then
    echo "Error: CSV file not found at $DIRMAP_PATH"
    exit 1
fi

if [ ! -d "$DATASET_PATH" ]; then
    echo "Error: Dataset directory not found at $DATASET_PATH"
    exit 1
fi

if [ ! -d "$MODEL_CKPT_DIR" ]; then
    echo "Error: Model checkpoint directory not found at $MODEL_CKPT_DIR"
    exit 1
fi

# Extract the base directory name for the dataset
base_dir=$(basename "$DATASET_PATH")

# Remove "imagenet16_" prefix if present
dataset_type=${base_dir#imagenet16_}

echo "Processing dataset: $base_dir"
echo "Dataset type: $dataset_type"

# Process each model checkpoint in MODEL_CKPT_DIR
for checkpoint in "$MODEL_CKPT_DIR"/*_checkpoint.pt; do
    if [ -f "$checkpoint" ]; then
        # Extract the checkpoint number
        checkpoint_basename=$(basename "$checkpoint")
        checkpoint_num=${checkpoint_basename%_checkpoint.pt}
        
        echo "--------------------------------"
        echo "Processing checkpoint: $checkpoint_basename (epoch: $checkpoint_num)"
        
        # Process each split (test, val, train)
        for split_dir in "$DATASET_PATH"/*; do
            if [ -d "$split_dir" ]; then
                split=$(basename "$split_dir")
                
                # Check if the split is one of the expected values
                if [[ "$split" == "test" || "$split" == "val" || "$split" == "train" ]]; then
                    echo "Processing split: $split"
                    
                    # Run the Python script with the specified parameters
                    python scripts/test_model_on_dirmap_get_gt_logit.py \
                        --dirmap_path "$DIRMAP_PATH" \
                        --dataset_name ImageNet \
                        --class_num_col orig_class_num \
                        --dataset_path "$DATASET_PATH" \
                        --model_ckpt_path "$checkpoint" \
                        --split "$split" \
                        --gt_logit_col_name "epoch${checkpoint_num}_gt_logit" \
                        --gpu_id "$GPU_ID"
                    
                    # Check if the Python script executed successfully
                    if [ $? -ne 0 ]; then
                        echo "Error: Python script failed for checkpoint epoch $checkpoint_num - $split"
                    fi
                fi
            fi
        done
    fi
done

echo "--------------------------------"
echo "Processing complete for all checkpoints!"