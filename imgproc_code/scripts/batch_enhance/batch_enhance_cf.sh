#!/bin/bash

# Dataset and model settings
# DATASET_NAME="MHIST"
# DATASET_PATH="/media/KLAB37/morgan_data/mhist/robustness"
# DIRMAP_CSV_PATH="/media/KLAB37/morgan_data/mhist/robustness/dirmap_test.csv"
# MODEL_PATH="train_output/82953832-1bbe-4648-8fac-0935ae54de3d/checkpoint.pt.best"

# Dataset and model settings
DATASET_NAME="HAM10000"
DATASET_PATH="/media/KLAB39/morgan_data/HAM10000"
DIRMAP_CSV_PATH="/media/KLAB39/morgan_data/HAM10000/ham10000_df_ham4.csv"
MODEL_PATH="model_ckpts/9c8ae3f6-2307-4d29-ae7a-e98095dd18bc/checkpoint.pt.best"

# Control variables
GPU_ID=3
DO_ATTACK=true  # Set to true to run the attack version
SAVE_ORIGINALS=true  # Set to true to include --save_originals option
CREATE_NEW_BUCKET=false

# Convert DATASET to lower case
DATASET_LOWER=$(echo "$DATASET_NAME" | tr '[:upper:]' '[:lower:]')

# Set save_orig based on SAVE_ORIGINALS
if $SAVE_ORIGINALS; then
    save_orig="--save_originals"
else
    save_orig=""
fi

if $CREATE_NEW_BUCKET; then
    create_new_bucket="--create_new_bucket"
else
    create_new_bucket=""
fi

# Get unique classes
get_unique_classes() {
    local csv_path="$1"
    # Assuming 'class' is the second column. Adjust the number if it's different.
    awk -F',' 'NR>1 {print $2}' "$csv_path" | sort -u
}
mapfile -t unique_classes < <(get_unique_classes "$DIRMAP_CSV_PATH")

for class in "${unique_classes[@]}"; do
    custom_eps=(4)   # Specify the set of epsilons like (4 8 12). For a sequence, use something like "for eps in $(seq 4 4 20); do"
    for eps in "${custom_eps[@]}"; do
        num_steps=$(printf "%.0f" $(echo "2 * $eps" | bc))   # Calculate the number of steps as 2x epsilon. This makes sure that the number of steps is an integer.

        CUDA_VISIBLE_DEVICES=${GPU_ID}, python enhance.py --eps $eps --num_steps $num_steps --dest_dir data/${DATASET_NAME} --dirmap_path ${DIRMAP_CSV_PATH} --dataset_name ${DATASET_NAME} --dataset_path ${DATASET_PATH} --model_ckpt_path ${MODEL_PATH} --objective_type logit_diverge ${save_orig} --diverge_from ${class}

        if $DO_ATTACK; then
            CUDA_VISIBLE_DEVICES=${GPU_ID}, python enhance.py --eps $eps --num_steps $num_steps --dest_dir data/${DATASET_NAME} --dirmap_path ${DIRMAP_CSV_PATH} --dataset_name ${DATASET_NAME} --dataset_path ${DATASET_PATH} --model_ckpt_path ${MODEL_PATH} --objective_type logit_diverge --attack --diverge_from ${class}
        fi

        save_orig="" # Stop saving originals after first round
        eps_dot=$(echo $eps | sed 's/\./dot/')

        python dataset_setup/upload_images_s3.py --data_path data/${DATASET_NAME}/${DATASET_NAME}_${eps}_0.5_${num_steps}_logit_diverge_from_${class} ${create_new_bucket} --bucket_name morgan-${DATASET_LOWER}-logit-diverge
        
        create_new_bucket="" # Don't need to create bucket anymore after first round

        if $DO_ATTACK; then
            python dataset_setup/upload_images_s3.py --data_path data/${DATASET_NAME}/${DATASET_NAME}_${eps}_0.5_${num_steps}_logit_diverge_from_${class}_attack --bucket_name morgan-${DATASET_LOWER}-logit-diverge
        fi
    done
done

if $SAVE_ORIGINALS; then
    natural_data_path=data/${DATASET_NAME}/${DATASET_NAME}_natural
    if [ ! -d natural_data_path ]; then
        CUDA_VISIBLE_DEVICES=${GPU_ID}, python enhance.py --save_originals --eps 0 --num_steps 0 --dest_dir data/${DATASET_NAME} --dirmap_path ${DIRMAP_CSV_PATH} --dataset_name ${DATASET_NAME} --dataset_path ${DATASET_PATH} --model_ckpt_path ${MODEL_PATH}
    fi
    python dataset_setup/upload_images_s3.py --data_path ${natural_data_path} --bucket_name morgan-${DATASET_LOWER}-logit-diverge
fi
