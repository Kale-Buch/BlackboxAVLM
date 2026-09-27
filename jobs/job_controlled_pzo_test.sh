#!/bin/bash
#SBATCH --job-name=controlled_pzo
#SBATCH --output=slurm_logs/controlled_pzo.out
#SBATCH --error=slurm_logs/controlled_pzo.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:nvidia_h100_nvl:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

cd /mnt/DISAPP/home/kalebuch/AdversarialVLM
export PYTHONPATH=$PYTHONPATH:$(pwd)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm

# ── CONTROLLED P-ZO PROOF TEST ────────────────────────────────────────────────
# 1. ZO_EVERY_N=1 (Victim feedback queried at every single iteration)
# 2. ZO_WEIGHT_MULT=5.0 (ZO gradient receives 5x weight in gradient aggregation)
# 3. Synchronized prompt across all 3 surrogates + victim per iteration
# 4. 150 iterations, tracking fixed-prompt victim loss every 25 steps
# 5. Fast victim evaluation on 10 test prompts via quick_zo_eval.py
# ──────────────────────────────────────────────────────────────────────────────

export MODEL_NAME="microsoft/Phi-3.5-vision-instruct,alpindale/Llama-3.2-11B-Vision-Instruct,llava-hf/llava-1.5-7b-hf"
export VICTIM="Qwen/Qwen2-VL-2B-Instruct"
export VICTIM_SUFFIX="qwenVL"
export EXP_NAME="controlled_pzo_eps005"
export NUM_ITERATIONS=150
export SAVE_STEPS=25
export GRAD_ACCUM_STEPS=1
export TRAIN_ONLY=1

export USE_QUERY_BASED="--use_query_based_gradient_check"
export USE_PZO="--use_pzo"
export PZO_EPSILON=0.05
export ZO_EVERY_N=1
export ZO_WEIGHT_MULT=5.0

echo "================================================================"
echo "=== CONTROLLED P-ZO TEST: RUNNING OPTIMIZATION ==="
echo "Surrogates : $MODEL_NAME"
echo "Victim     : $VICTIM"
echo "Iterations : $NUM_ITERATIONS"
echo "ZO Every N : $ZO_EVERY_N (Continuous feedback)"
echo "ZO Weight  : $ZO_WEIGHT_MULT (Dominant signal)"
echo "Prompts    : Synchronized across all 4 models per step"
echo "================================================================"

./scripts/attacks/attack_blackbox_enhanced.sh

EXP_PATH=$(ls -td runs/controlled_pzo_sync_* | head -n 1)

echo "================================================================"
echo "=== FIXED-PROMPT VICTIM LOSS TRAJECTORY (victim_zo_log.csv) ==="
echo "================================================================"
cat "$EXP_PATH/victim_zo_log.csv"

echo "================================================================"
echo "=== DETERMINISTIC MULTI-PROMPT EVALUATION (quick_zo_eval.py) ==="
echo "================================================================"
python3 src/quick_zo_eval.py --exp_path "$EXP_PATH" --victim "$VICTIM" --device "cuda:0" --num_questions 10

echo "================================================================"
echo "=== CONTROLLED P-ZO TEST COMPLETED ==="
echo "================================================================"
