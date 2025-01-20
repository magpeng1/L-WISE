#!/bin/bash

# Dataset and model settings
DATASET_NAME_RB="HAM10000"
DATASET_NAME="HAM10000"
DATASET_PATH="/media/KLAB39/morgan_data/HAM10000"
DIRMAP_CSV_PATH="/media/KLAB39/morgan_data/HAM10000/ham10000_df_ham4.csv"
MODEL_PATH="model_ckpts/HAM10000_eps1.pt"


# Control variables
GPU_ID=0
DO_ATTACK=false  # Set to true to run the attack version
SAVE_ORIGINALS=false  # Set to true to include --save_originals option

# Convert DATASET to lower case
DATASET_LOWER=$(echo "$DATASET_NAME" | tr '[:upper:]' '[:lower:]')

# Set save_orig based on SAVE_ORIGINALS
if $SAVE_ORIGINALS; then
    save_orig="--save_originals"
else
    save_orig=""
fi

custom_eps=(0.25 0.5 1 2 4 8)
for eps in "${custom_eps[@]}"; do
    num_steps=$(printf "%.0f" $(echo "2 * $eps" | bc))   # Calculate the number of steps as 2x epsilon. This makes sure that the number of steps is an integer.

    if [ "$num_steps" -eq 0 ]; then
        echo "num_steps evaluated to 0, setting it to 1 instead"
        num_steps=1
    fi

    CUDA_VISIBLE_DEVICES=${GPU_ID}, python enhance.py --eps $eps --num_steps $num_steps --dest_dir data/${DATASET_NAME} --dirmap_path ${DIRMAP_CSV_PATH} --dataset_name ${DATASET_NAME_RB} --dataset_path ${DATASET_PATH} --model_ckpt_path ${MODEL_PATH} --objective_type logit_diverge ${save_orig}

    if $DO_ATTACK; then
        CUDA_VISIBLE_DEVICES=${GPU_ID}, python enhance.py --eps $eps --num_steps $num_steps --dest_dir data/${DATASET_NAME} --dirmap_path ${DIRMAP_CSV_PATH} --dataset_name ${DATASET_NAME_RB} --dataset_path ${DATASET_PATH} --model_ckpt_path ${MODEL_PATH} --objective_type logit_diverge --attack
    fi

    save_orig=""
    eps_dot=$(echo $eps | sed 's/\./dot/')

    python dataset_setup/upload_images_s3.py --data_path data/${DATASET_NAME}/${DATASET_NAME}_${eps}_0.5_${num_steps}_logit_diverge --create_new_bucket --bucket_name morgan-${DATASET_LOWER}-${eps_dot}-0dot5-${num_steps}-logit-diverge
    
    if $DO_ATTACK; then
        python dataset_setup/upload_images_s3.py --data_path data/${DATASET_NAME}/${DATASET_NAME}_${eps}_0.5_${num_steps}_logit_diverge_attack --create_new_bucket --bucket_name morgan-${DATASET_LOWER}-${eps}-0dot5-${num_steps}-logit-diverge-attack
    fi
done

if $SAVE_ORIGINALS; then
    python dataset_setup/upload_images_s3.py --data_path data/${DATASET_NAME}/${DATASET_NAME}_natural --create_new_bucket --bucket_name morgan-${DATASET_LOWER}-natural
fi
