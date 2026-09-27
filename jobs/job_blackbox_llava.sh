#!/bin/bash
#SBATCH --job-name=gpu_black_llava
#SBATCH --output=slurm_logs/gpu_black_llava.out
#SBATCH --error=slurm_logs/gpu_black_llava.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:nvidia_h100_nvl:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

cd /mnt/DISAPP/home/kalebuch/AdversarialVLM
export PYTHONPATH=$PYTHONPATH:$(pwd)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm

# ── Victim (blackbox query-only) ───────────────────────────────────────────────
export VICTIM="llava-hf/llava-1.5-7b-hf"
export VICTIM_SUFFIX="llava-hf"

# ── Surrogates (white-box, must exclude victim) ────────────────────────────────
export MODEL_NAME="microsoft/Phi-3.5-vision-instruct,alpindale/Llama-3.2-11B-Vision-Instruct,Qwen/Qwen2-VL-2B-Instruct"

export EXP_NAME="llava_atk_enhanced_transfer"

./scripts/attacks/attack_blackbox_enhanced.sh
