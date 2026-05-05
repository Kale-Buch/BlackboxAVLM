#!/bin/bash

# Initialize conda for the script
source ~/miniconda3/etc/profile.d/conda.sh


# Configuration
EXP_NAME="automated_transfer_test"
IMG_ORIG="gray.png"
PROMPT="list"
TARGET_TEXT="sure, here it is"
MODEL_NAME="llava-hf/llava-1.5-7b-hf,alpindale/Llama-3.2-11B-Vision-Instruct,Qwen/Qwen2-VL-2B-Instruct"
VICTIM="microsoft/Phi-3.5-vision-instruct"
VICTIM_SUFFIX="phi35" # corresponds to the dictionary in SafeBench_universal.py

# Default CUDA devices (can be overridden by environment)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
CUDA_NUM=0 # Device to use for evaluation

LR=1e-2
ATTACK_NORM=0.4
NUM_ITERATIONS=10000
SAVE_STEPS=100
BATCH_SIZE=1
GRAD_ACCUM_STEPS=1
CLAMP_METHOD="tanh"
RESTART_NUM=0

# File containing the WandB API key
WANDB_KEY_FILE="wandb_key.txt"
if [ ! -f "$WANDB_KEY_FILE" ]; then
    echo "Error: WandB key file '$WANDB_KEY_FILE' not found."
    exit 1
fi
export WANDB_API_KEY=$(cat <"$WANDB_KEY_FILE")

# Path for PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Temporary file to capture experiment path
OUT_PATH_FILE="tmp_exp_path.txt"
touch "$OUT_PATH_FILE"

echo "================================================"
echo "=== STEP 1: Starting Cross-Attack Training ==="
echo "Models: $MODEL_NAME"
echo "Environment: vlm"
echo "================================================"

# Activate VLM environment for surrogates
conda activate vlm

# Run Training
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 "src/crossattack_models.py" \
    --exp_name "$EXP_NAME" \
    --img_orig "$IMG_ORIG" \
    --prompt "$PROMPT" \
    --target_text "$TARGET_TEXT" \
    --model_name "$MODEL_NAME" \
    --lr "$LR" \
    --num_iterations "$NUM_ITERATIONS" \
    --save_steps "$SAVE_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum_steps "$GRAD_ACCUM_STEPS" \
    --scheduler_step_size 100 \
    --scheduler_gamma 1.0 \
    --clamp_method "$CLAMP_METHOD" \
    --restart_num "$RESTART_NUM" \
    --epsilon "$ATTACK_NORM" \
    --output_path_file "$OUT_PATH_FILE" \
    --use_gaussian_blur \
    --gblur_kernel_size 5


if [ ! -s "$OUT_PATH_FILE" ]; then
    echo "Error: Training failed to produce output path file or it is empty."
    rm "$OUT_PATH_FILE"
    exit 1
fi

EXP_PATH=$(cat "$OUT_PATH_FILE")
rm "$OUT_PATH_FILE"
UNIQUE_EXP_NAME=$(basename "$EXP_PATH")
echo "Experiment directory successfully captured: $EXP_PATH"
echo "Unique experiment name: $UNIQUE_EXP_NAME"

echo "================================================"
echo "=== STEP 2: Finding Best Iteration (Gemma Judge) ==="
echo "Environment: vlmg"
echo "================================================"

# Activate VLMG environment for Gemma judge
conda activate vlmg

# Find best iteration using safety judge
./scripts/evaluation/find_best_iter.sh

# Extract best iteration
if [ -f "$EXP_PATH/best.txt" ]; then
    BEST_ITER=$(cat "$EXP_PATH/best.txt")
else
    # Fallback if best.txt is not at root
    BEST_ITER=$(cat "$EXP_PATH/safety_details/gemma-3-4b-it/best_iter.txt")
fi

echo "Found Best Iteration: $BEST_ITER"

echo "================================================"
echo "=== STEP 3: Evaluating Transfer on Victim Model (SafeBench) ==="
echo "Victim: $VICTIM ($VICTIM_SUFFIX)"
echo "Environment: vlm"
echo "================================================"

# Activate VLM environment for running SafeBench evaluation
conda activate vlm

# 2. Test on SafeBench dataset
./scripts/evaluation/safebench_test.sh "$UNIQUE_EXP_NAME" "$BEST_ITER" "$VICTIM_SUFFIX" "$CUDA_NUM"

echo "================================================"
echo "=== STEP 4: Final Safety Check on SafeBench Results ==="
echo "Environment: vlmg"
echo "================================================"

# Calculate results path
RESULTS_PATH="tests/${UNIQUE_EXP_NAME}_${BEST_ITER}/${VICTIM_SUFFIX}"

# 3. Evaluate with safety judge
conda activate vlmg
./scripts/evaluation/guard_eval.sh "$RESULTS_PATH" "$CUDA_NUM"

echo "================================================"
echo "=== PIPELINE COMPLETED SUCCESSFULLY ==="
echo "================================================"
