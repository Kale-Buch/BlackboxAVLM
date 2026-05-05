#!/bin/bash

# Configuration for FAST TEST
EXP_NAME="test_handshake"
IMG_ORIG="gray.png"
PROMPT="list"
TARGET_TEXT="sure, here it is"
# Minimal surrogates to speed up loading
SURROGATES="microsoft/Phi-3.5-vision-instruct"
VICTIM="Qwen/Qwen2-VL-2B-Instruct"

LR=1e-2
ATTACK_NORM=0.4
NUM_ITERATIONS=1
SAVE_STEPS=1
BATCH_SIZE=1
GRAD_ACCUM_STEPS=1

# File containing the WandB API key
WANDB_KEY_FILE="wandb_key.txt"
export WANDB_API_KEY=$(cat <"$WANDB_KEY_FILE")
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Temporary file to capture experiment path
OUT_PATH_FILE="tmp_test_path.txt"
touch "$OUT_PATH_FILE"

echo "=== STEP 1: FAST TRAINING (1 ITER) ==="
python3 "src/crossattack_models.py" \
    --exp_name "$EXP_NAME" \
    --img_orig "$IMG_ORIG" \
    --prompt "$PROMPT" \
    --target_text "$TARGET_TEXT" \
    --model_name "$SURROGATES" \
    --lr "$LR" \
    --num_iterations "$NUM_ITERATIONS" \
    --save_steps "$SAVE_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum_steps "$GRAD_ACCUM_STEPS" \
    --epsilon "$ATTACK_NORM" \
    --output_path_file "$OUT_PATH_FILE"

if [ ! -s "$OUT_PATH_FILE" ]; then
    echo "Error: Training failed."
    rm "$OUT_PATH_FILE"
    exit 1
fi

EXP_PATH=$(cat "$OUT_PATH_FILE")
rm "$OUT_PATH_FILE"
echo "Experiment directory: $EXP_PATH"

echo "=== STEP 2: BEST ITERATION (GEMMA) ==="
# We only want to process the test directory
# (find_best_iter_gemma will scan runs/, so this is fine)
python3 src/evaluation/find_best_iter_gemma.py

echo "=== STEP 3: TRANSFER EVAL (QWEN) ==="
# Test on only 2 questions to save time
# (We need to modify evaluate_transfer temporarily or just let it run)
# I'll let it run but with a small question set if I can.
python3 src/evaluation/evaluate_transfer.py \
    --exp_path "$EXP_PATH" \
    --victim_model "$VICTIM" \
    --target_text "$TARGET_TEXT"
