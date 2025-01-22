#!/bin/bash

# VALIDATION SET Dataset and model settings
DATASET_NAME_RB="ImageNet"  # This is the name of the dataset as seen in the robustness library (e.g., ImageNet)
DATASET_NAME="imagenet16_xcit_large_eps4_aug"  # This is the name of the dataset as we will actually name it (e.g., imagenet16)
DATASET_PATH="/home/morgan/projects/learn-histopath-backend/data/imagenet16"
DIRMAP_CSV_PATH="/home/morgan/projects/learn-histopath-backend/data/imagenet16/dirmap.csv"
MODEL_PATH="model_ckpts/debenedetti/xcit-l12-ImageNet-eps-4.pth.tar"
MODEL_ARCH="xcit_large_12_p16_224"

# Control variables
GPU_ID=0
DO_ATTACK=false  # Set to true to run the attack version
SAVE_ORIGINALS=false  # Set to true to include --save_originals option
BATCH_SIZE=4
NUM_AUGS=4

# Convert DATASET to lower case
DATASET_LOWER=$(echo "$DATASET_NAME" | tr '[:upper:]' '[:lower:]')

# Set save_orig based on SAVE_ORIGINALS
if $SAVE_ORIGINALS; then
    save_orig="--save_originals"
else
    save_orig=""
fi
 
#custom_eps=(5 10 15 20) # Specify the set of epsilons. For a sequence, use something like "for eps in $(seq 4 4 20); do"
custom_eps=(20)
for eps in "${custom_eps[@]}"; do
    num_steps=$(printf "%.0f" $(echo "2 * $eps" | bc))   # Calculate the number of steps as 2x epsilon. This makes sure that the number of steps is an integer.

    if [ "$num_steps" -eq 0 ]; then
        echo "num_steps evaluated to 0, setting it to 1 instead"
        num_steps=1
    fi

    CUDA_VISIBLE_DEVICES=${GPU_ID}, python enhance_vit_aug.py --arch $MODEL_ARCH --eps $eps --num_steps $num_steps --num_augs $NUM_AUGS --batch_size $BATCH_SIZE --dest_dir data/${DATASET_NAME} --dirmap_path ${DIRMAP_CSV_PATH} --dataset_name ${DATASET_NAME_RB} --dataset_path ${DATASET_PATH} --model_ckpt_path ${MODEL_PATH} --objective_type logit ${save_orig}

    if $DO_ATTACK; then
        CUDA_VISIBLE_DEVICES=${GPU_ID}, python enhance_vit_aug.py --arch $MODEL_ARCH --eps $eps --num_steps $num_steps --num_augs $NUM_AUGS --batch_size $BATCH_SIZE --dest_dir data/${DATASET_NAME} --dirmap_path ${DIRMAP_CSV_PATH} --dataset_name ${DATASET_NAME_RB} --dataset_path ${DATASET_PATH} --model_ckpt_path ${MODEL_PATH} --objective_type logit --attack
    fi

    save_orig=""
    eps_dot=$(echo $eps | sed 's/\./dot/')

    # python dataset_setup/upload_images_s3.py --data_path data/${DATASET_NAME}/${DATASET_NAME}_${eps}_0.5_${num_steps}_logit_diverge --create_new_bucket --bucket_name morgan-${DATASET_LOWER}-full-${eps_dot}-0dot5-${num_steps}-logit
    
    # if $DO_ATTACK; then
    #     python dataset_setup/upload_images_s3.py --data_path data/${DATASET_NAME}/${DATASET_NAME}_${eps}_0.5_${num_steps}_logit_diverge_attack --create_new_bucket --bucket_name morgan-${DATASET_LOWER}-full-${eps}-0dot5-${num_steps}-logit-attack
    # fi
done

if $SAVE_ORIGINALS; then
    python dataset_setup/upload_images_s3.py --data_path data/${DATASET_NAME}/${DATASET_NAME}_natural --create_new_bucket --bucket_name morgan-${DATASET_LOWER}-full-natural
fi
