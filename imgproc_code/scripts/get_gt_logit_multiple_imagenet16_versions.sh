#!/bin/bash

# Check if required arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 DIRMAP_PATH DATA_PATH [MODEL_CKPT_PATH] [GPU_ID]"
    echo "DIRMAP_PATH: Path to the CSV file"
    echo "DATA_PATH: Path to the directory containing model directories"
    echo "MODEL_CKPT_PATH: Optional path to model checkpoint (default: model_ckpts/ImageNet_eps3.pt)"
    echo "GPU_ID: Optional GPU ID (default: 0)"
    exit 1
fi

# Assign arguments to variables
DIRMAP_PATH="$1"
DATA_PATH="$2"
MODEL_CKPT_PATH="${3:-model_ckpts/ImageNet_eps3.pt}"  # Set default if not provided
GPU_ID="${4:-0}"  # Set default GPU_ID to 0 if not provided

# Check if the paths exist
if [ ! -f "$DIRMAP_PATH" ]; then
    echo "Error: CSV file not found at $DIRMAP_PATH"
    exit 1
fi

if [ ! -d "$DATA_PATH" ]; then
    echo "Error: Directory not found at $DATA_PATH"
    exit 1
fi

# Process each directory in DATA_PATH
for dir in "$DATA_PATH"/*; do
    if [ -d "$dir" ]; then
        # Extract the base directory name
        base_dir=$(basename "$dir")
        
        # Remove "imagenet16_" prefix if present
        model_type=${base_dir#imagenet16_}
        
        echo "Processing model type: $model_type"
        echo "Location of dataset: $dir"
        
        # Process each split (test, val, train)
        for split_dir in "$dir"/*; do
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
                        --dataset_path "$dir" \
                        --model_ckpt_path $MODEL_CKPT_PATH \
                        --split "$split" \
                        --gt_logit_col_name "${model_type}_robust_gt_logit" \
                        --gpu_id "$GPU_ID"
                    
                    # Check if the Python script executed successfully
                    if [ $? -ne 0 ]; then
                        echo "Error: Python script failed for $model_type - $split"
                    fi
                fi
            fi
        done
    fi
done

echo "Processing complete!"