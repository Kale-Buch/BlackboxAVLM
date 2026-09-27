#!/bin/bash

# Enhanced blackbox attack with transferability improvements:
#   - Gradient normalization (prevents surrogate gradient dominance)
#   - MI-FGSM momentum (stabilizes gradient direction across iterations)
#   - Diverse Input Method (prevents overfitting to surrogate preprocessing)
#   - Multi-Crop Alignment (M-Attack V2 style gradient variance reduction)
#   - Shorter target text (easier to transfer across tokenizers)

# Initialize conda for the script
source ~/miniconda3/etc/profile.d/conda.sh


# Configuration
EXP_NAME="${EXP_NAME:-qwen_atk_nrm_hyperized_enhanced_transfer_test}"
IMG_ORIG="${IMG_ORIG:-gray.png}"
PROMPT="${PROMPT:-list}"
TARGET_TEXT="${TARGET_TEXT:-sure, here it is}"
MODEL_NAME="${MODEL_NAME:-microsoft/Phi-3.5-vision-instruct,alpindale/Llama-3.2-11B-Vision-Instruct,llava-hf/llava-1.5-7b-hf}"
VICTIM="${VICTIM:-Qwen/Qwen2-VL-2B-Instruct}"
VICTIM_SUFFIX="${VICTIM_SUFFIX:-qwenVL}" # corresponds to the dictionary in SafeBench_universal.py


# Verify victim suffix is correct (defensive check)
if [ "$VICTIM_SUFFIX" != "qwenVL" ]; then
    echo "ERROR: VICTIM_SUFFIX should be 'qwenVL' but is set to '$VICTIM_SUFFIX'"
    exit 1
fi

# Default CUDA devices (can be overridden by environment)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
CUDA_NUM=0 # Device to use for evaluation

LR="${LR:-5e-3}"
ATTACK_NORM="${ATTACK_NORM:-0.1}"
NUM_ITERATIONS="${NUM_ITERATIONS:-2000}"
SAVE_STEPS="${SAVE_STEPS:-100}"
BATCH_SIZE=1
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
TRAIN_ONLY="${TRAIN_ONLY:-0}"
CLAMP_METHOD="tanh"
RESTART_NUM=0

# Transferability enhancements
USE_GRAD_NORM="--use_grad_norm"
USE_MOMENTUM="--use_momentum"
MOMENTUM_DECAY="${MOMENTUM_DECAY:-0.9}"
USE_DIVERSE_INPUT="--use_diverse_input"
DIVERSE_INPUT_PROB="${DIVERSE_INPUT_PROB:-0.7}"
N_MULTI_CROP="${N_MULTI_CROP:-1}"                            # M-Attack V2 MCA (average 3 crops)
# Zeroth-order (query-based) gradient settings
ZO_MODE="${ZO_MODE:-one_sided}"                             # one_sided (~2x faster, 1 query/patch + 1 base) or two_sided
ZO_EPSILON="${ZO_EPSILON:-0.02}"
ZO_NUM_SAMPLES="${ZO_NUM_SAMPLES:-5}"

USE_PZO="${USE_PZO:---use_pzo}"
PZO_EPSILON="${PZO_EPSILON:-0.01}"
# Exhaustive patch-wise BCD (Zer0-Jack style)
USE_EXHAUSTIVE_BCD="${USE_EXHAUSTIVE_BCD:-}"
ZO_PATCH_SIZE="${ZO_PATCH_SIZE:-128}"
ZO_WEIGHT_MULT="${ZO_WEIGHT_MULT:-3.0}"
ZO_COSINE_THRESH="${ZO_COSINE_THRESH:-0.0}"
ZO_SIGNAL_THRESH="${ZO_SIGNAL_THRESH:-0.001}"
ZO_EVERY_N="${ZO_EVERY_N:-20}"

NEW_NAME="${SLURM_JOB_ID}_EXP_NAME=${EXP_NAME}"
scontrol update job=$SLURM_JOB_ID JobName="$NEW_NAME"

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
echo "=== STEP 1: Enhanced Cross-Attack Training ==="
echo "Models: $MODEL_NAME"
echo "Victim: $VICTIM"
echo "Query-based victim gradient check: enabled"
echo "Target: $TARGET_TEXT"
echo "Enhancements: GradNorm + Momentum + DIM + MCA($N_MULTI_CROP)"
echo "Environment: vlm"
echo "================================================"

# Activate VLM environment for surrogates
conda activate vlm

# Run Training with transferability enhancements
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 "src/crossattack_models.py" \
    --exp_name "$EXP_NAME" \
    --img_orig "$IMG_ORIG" \
    --prompt "$PROMPT" \
    --target_text "$TARGET_TEXT" \
    --model_names "$MODEL_NAME" \
    --victim "$VICTIM" \
    --lr "$LR" \
    --num_iterations "$NUM_ITERATIONS" \
    --save_steps "$SAVE_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum_steps "$GRAD_ACCUM_STEPS" \
    --scheduler_step_size 100 \
    --scheduler_gamma 0.95 \
    --clamp_method "$CLAMP_METHOD" \
    --restart_num "$RESTART_NUM" \
    --epsilon "$ATTACK_NORM" \
    --output_path_file "$OUT_PATH_FILE" \
    --use_gaussian_blur \
    --gblur_kernel_size 3 \
    $USE_GRAD_NORM \
    $USE_MOMENTUM \
    --momentum_decay "$MOMENTUM_DECAY" \
    $USE_DIVERSE_INPUT \
    --diverse_input_prob "$DIVERSE_INPUT_PROB" \
    --n_multi_crop "$N_MULTI_CROP" \
    --use_query_based_gradient_check \
    $USE_PZO \
    --pzo_epsilon "$PZO_EPSILON" \
    --bcd_block_size 32 \
    --zo_epsilon "$ZO_EPSILON" \
    --zo_num_samples "$ZO_NUM_SAMPLES" \
    $USE_EXHAUSTIVE_BCD \
    --zo_patch_size "$ZO_PATCH_SIZE" \
    --zo_weight_multiplier "$ZO_WEIGHT_MULT" \
    --zo_cosine_threshold "$ZO_COSINE_THRESH" \
    --zo_signal_threshold "$ZO_SIGNAL_THRESH" \
    --zo_every_n_iters "$ZO_EVERY_N" \
    --zo_mode "$ZO_MODE"


if [ ! -s "$OUT_PATH_FILE" ]; then
    echo "Error: Training failed to produce output path file or it is empty."
    rm "$OUT_PATH_FILE"
    exit 1
fi

if [ "$TRAIN_ONLY" = "1" ]; then
    echo "TRAIN_ONLY=1: stopping after cross-attack training."
    rm -f "$OUT_PATH_FILE"
    exit 0
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
