#!/bin/bash
#SBATCH --job-name=gpu_black_qwen
#SBATCH --output=slurm_logs/gpu_black_qwen.out
#SBATCH --error=slurm_logs/gpu_black_qwen.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:nvidia_h100_nvl:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

cd /mnt/DISAPP/home/kalebuch/AdversarialVLM
export PYTHONPATH=$PYTHONPATH:$(pwd)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm

# ── Victim (blackbox query-only) ───────────────────────────────────────────────
export VICTIM="Qwen/Qwen2-VL-2B-Instruct"
export VICTIM_SUFFIX="qwenVL"

# ── Surrogates (white-box, must exclude victim) ────────────────────────────────
export MODEL_NAME="microsoft/Phi-3.5-vision-instruct,alpindale/Llama-3.2-11B-Vision-Instruct,llava-hf/llava-1.5-7b-hf"

export EXP_NAME="qwen_atk_enhanced_transfer_500iter"
export NUM_ITERATIONS=500
export SAVE_STEPS=100

./scripts/attacks/attack_blackbox_enhanced.sh

