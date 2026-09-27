#!/bin/bash

# Generic enhanced blackbox cross-model attack.
#
# All victim and surrogate configuration is driven by environment variables —
# no model names or architecture assumptions are hardcoded here.
#
# Transferability techniques:
#   - Gradient normalization  (prevents any single surrogate from dominating)
#   - MI-FGSM momentum        (stabilises gradient direction across iterations)
#   - Diverse Input Method    (prevents overfitting to surrogate preprocessors)
#   - Multi-Crop Alignment    (M-Attack V2 style gradient variance reduction)
#   - Exhaustive patch-wise ZO (Zer0-Jack style query-based victim gradient)
#
# BLACKBOX CONSTRAINT: the victim model is NEVER used for white-box gradients.
# The only victim access is zeroth-order queries (loss evaluations).
# Set USE_QUERY_BASED="" to disable ZO entirely (pure surrogate transfer).

# Initialize conda for the script
source ~/miniconda3/etc/profile.d/conda.sh

# ── Victim & surrogate configuration ──────────────────────────────────────────
# Override all of these from your job file.  No defaults enforce a specific
# architecture — the caller is responsible for setting a consistent pair.
EXP_NAME="${EXP_NAME:-blackbox_enhanced_transfer}"
IMG_ORIG="${IMG_ORIG:-gray.png}"
PROMPT="${PROMPT:-list}"
TARGET_TEXT="${TARGET_TEXT:-sure, here it is}"   # Short target = better cross-tokenizer transfer

# Surrogate models (comma-separated HuggingFace IDs, must exclude victim)
MODEL_NAME="${MODEL_NAME:?ERROR: MODEL_NAME must be set by the calling job (surrogate model IDs, comma-separated)}"

# Victim model (HuggingFace ID) — queried blackbox only
VICTIM="${VICTIM:?ERROR: VICTIM must be set by the calling job (victim model HF ID)}"

# Victim suffix — must match the key in SafeBench_universal.py dict_suf_modelname
# Valid values: qwenVL | phi35 | Llama32 | llava-hf
VICTIM_SUFFIX="${VICTIM_SUFFIX:?ERROR: VICTIM_SUFFIX must be set by the calling job}"

# ── CUDA ───────────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
CUDA_NUM=0    # Device used for SafeBench evaluation (Step 3)

# ── Training hyperparameters ───────────────────────────────────────────────────
LR="${LR:-5e-3}"
ATTACK_NORM="${ATTACK_NORM:-0.2}"
NUM_ITERATIONS="${NUM_ITERATIONS:-5000}"
SAVE_STEPS="${SAVE_STEPS:-100}"
BATCH_SIZE=1
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
TRAIN_ONLY="${TRAIN_ONLY:-0}"
CLAMP_METHOD="tanh"
RESTART_NUM=0

# ── Surrogate-side transferability enhancements ────────────────────────────────
USE_GRAD_NORM="--use_grad_norm"
USE_MOMENTUM="--use_momentum"
MOMENTUM_DECAY="${MOMENTUM_DECAY:-0.9}"
USE_DIVERSE_INPUT="--use_diverse_input"
DIVERSE_INPUT_PROB="${DIVERSE_INPUT_PROB:-0.7}"
N_MULTI_CROP="${N_MULTI_CROP:-1}"

# ── Zeroth-order (query-based) victim gradient ─────────────────────────────────
# These flags control the ONLY interaction with the victim model.
# Set USE_QUERY_BASED="" to run pure surrogate-transfer with no victim queries.
USE_QUERY_BASED="${USE_QUERY_BASED:---use_query_based_gradient_check}"

ZO_MODE="${ZO_MODE:-one_sided}"         # one_sided (~2× faster) or two_sided
ZO_EPSILON="${ZO_EPSILON:-0.02}"
ZO_NUM_SAMPLES="${ZO_NUM_SAMPLES:-5}"

USE_PZO="${USE_PZO:---use_pzo}"
PZO_EPSILON="${PZO_EPSILON:-0.05}"
# Exhaustive patch-wise BCD (sweeps every patch; set USE_EXHAUSTIVE_BCD="--use_exhaustive_bcd" to enable)
USE_EXHAUSTIVE_BCD="${USE_EXHAUSTIVE_BCD:-}"
ZO_PATCH_SIZE="${ZO_PATCH_SIZE:-128}"
ZO_WEIGHT_MULT="${ZO_WEIGHT_MULT:-3.0}"
ZO_COSINE_THRESH="${ZO_COSINE_THRESH:-0.0}"
ZO_SIGNAL_THRESH="${ZO_SIGNAL_THRESH:-0.001}"
ZO_EVERY_N="${ZO_EVERY_N:-5}"

# ── Victim-Dominant Hybrid Attack (g_victim + surr_lambda * g_ens with P-RGF) ──
USE_VICTIM_DOMINANT="${USE_VICTIM_DOMINANT:-}"
SURR_LAMBDA="${SURR_LAMBDA:-0.1}"
PRGF_LAMBDA="${PRGF_LAMBDA:-0.5}"
PRGF_PATCH_SIZE="${PRGF_PATCH_SIZE:-32}"
PRGF_NUM_DIRS="${PRGF_NUM_DIRS:-15}"

# ── Bookkeeping ────────────────────────────────────────────────────────────────
NEW_NAME="${SLURM_JOB_ID}_EXP_NAME=${EXP_NAME}"
scontrol update job=$SLURM_JOB_ID JobName="$NEW_NAME" 2>/dev/null || true

WANDB_KEY_FILE="wandb_key.txt"
if [ ! -f "$WANDB_KEY_FILE" ]; then
    echo "Error: WandB key file '$WANDB_KEY_FILE' not found."
    exit 1
fi
export WANDB_API_KEY=$(cat <"$WANDB_KEY_FILE")

export PYTHONPATH=$PYTHONPATH:$(pwd)

OUT_PATH_FILE="tmp_exp_path.txt"
touch "$OUT_PATH_FILE"

echo "================================================"
echo "=== STEP 1: Enhanced Cross-Attack Training ==="
echo "Surrogates : $MODEL_NAME"
echo "Victim     : $VICTIM  (suffix=$VICTIM_SUFFIX)"
echo "ZO queries : ${USE_QUERY_BASED:-(disabled)}"
echo "Victim-Dom : ${USE_VICTIM_DOMINANT:-(disabled)} (surr_lam=$SURR_LAMBDA, prgf_lam=$PRGF_LAMBDA, patch=$PRGF_PATCH_SIZE, dirs=$PRGF_NUM_DIRS)"
echo "ZO mode    : $ZO_MODE  patch=$ZO_PATCH_SIZE  weight=$ZO_WEIGHT_MULT  every_n=$ZO_EVERY_N"
echo "Target text: $TARGET_TEXT"
echo "Iterations : $NUM_ITERATIONS"
echo "================================================"

conda activate vlm

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
    $USE_QUERY_BASED \
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
    --zo_mode "$ZO_MODE" \
    ${USE_VICTIM_DOMINANT:+--use_victim_dominant} \
    --surr_lambda "$SURR_LAMBDA" \
    --prgf_lambda "$PRGF_LAMBDA" \
    --prgf_patch_size "$PRGF_PATCH_SIZE" \
    --prgf_num_dirs "$PRGF_NUM_DIRS"


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
echo "Experiment directory: $EXP_PATH"

echo "================================================"
echo "=== STEP 2: Finding Best Iteration (Gemma Judge) ==="
echo "================================================"

conda activate vlmg
./scripts/evaluation/find_best_iter.sh

# Prioritize victim best iteration in victim-dominant mode, otherwise surrogate-ASR checkpoint (best.txt).
if [ -n "$USE_VICTIM_DOMINANT" ] && [ -f "$EXP_PATH/best_victim.txt" ]; then
    BEST_ITER=$(cat "$EXP_PATH/best_victim.txt")
    echo "Using victim best iteration (victim-dominant mode): $BEST_ITER"
elif [ -f "$EXP_PATH/best.txt" ] && [ "$(cat "$EXP_PATH/best.txt")" != "0" ]; then
    BEST_ITER=$(cat "$EXP_PATH/best.txt")
    echo "Using surrogate best iteration (primary): $BEST_ITER"
elif [ -f "$EXP_PATH/best_victim.txt" ]; then
    BEST_ITER=$(cat "$EXP_PATH/best_victim.txt")
    echo "Using victim best iteration (fallback): $BEST_ITER"
elif [ -f "$EXP_PATH/best.txt" ]; then
    BEST_ITER=$(cat "$EXP_PATH/best.txt")
    echo "Using surrogate best iteration (fallback): $BEST_ITER"
else
    BEST_ITER=$(cat "$EXP_PATH/safety_details/gemma-3-4b-it/best_iter.txt")
    echo "Using safety-details best iteration (fallback): $BEST_ITER"
fi

echo "Found Best Iteration: $BEST_ITER"

echo "================================================"
echo "=== STEP 3: Evaluating Transfer on Victim (SafeBench) ==="
echo "Victim: $VICTIM  ($VICTIM_SUFFIX)"
echo "================================================"

conda activate vlm
./scripts/evaluation/safebench_test.sh "$UNIQUE_EXP_NAME" "$BEST_ITER" "$VICTIM_SUFFIX" "$CUDA_NUM"

echo "================================================"
echo "=== STEP 4: Final Safety Check on SafeBench Results ==="
echo "================================================"

RESULTS_PATH="tests/${UNIQUE_EXP_NAME}_${BEST_ITER}/${VICTIM_SUFFIX}"

conda activate vlmg
./scripts/evaluation/guard_eval.sh "$RESULTS_PATH" "$CUDA_NUM"

echo "================================================"
echo "=== PIPELINE COMPLETED SUCCESSFULLY ==="
echo "================================================"
