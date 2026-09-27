#!/bin/bash
#SBATCH --job-name=qwen_vdom
#SBATCH --output=slurm_logs/qwen_vdom_%j.out
#SBATCH --error=slurm_logs/qwen_vdom_%j.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:nvidia_h100_nvl:4
#SBATCH --exclude=rpg-93-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

cd /mnt/DISAPP/home/kalebuch/AdversarialVLM
export PYTHONPATH=$PYTHONPATH:$(pwd)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm

# ── Victim (blackbox query-only) ───────────────────────────────────────────────
export VICTIM="Qwen/Qwen2-VL-2B-Instruct"
export VICTIM_SUFFIX="qwenVL"

# ── Surrogates (white-box, must exclude victim) ────────────────────────────────
export MODEL_NAME="microsoft/Phi-3.5-vision-instruct,alpindale/Llama-3.2-11B-Vision-Instruct,llava-hf/llava-1.5-7b-hf"

export EXP_NAME="qwen_victim_dominant_full"

# ── Victim-Dominant Hybrid Parameters ─────────────────────────────────────────
export USE_VICTIM_DOMINANT="1"
export TARGET_TEXT="Sure"
export SURR_LAMBDA="0.1"
export PRGF_LAMBDA="0.5"
export PRGF_PATCH_SIZE="32"
export PRGF_NUM_DIRS="75"
export ZO_EPSILON="0.05"
export LR="1e-2"
export ATTACK_NORM="0.3"
export NUM_ITERATIONS="2000"
export SAVE_STEPS="25"

./scripts/attacks/attack_blackbox_enhanced.sh
