#!/bin/bash

# Submit a set of attack_blackbox_enhanced_qwen.sh variants with different hyperparameters.
# Usage: ./scripts/attacks/submit_qwen_variants.sh

source ~/miniconda3/etc/profile.d/conda.sh
cd /mnt/DISAPP/home/kalebuch/AdversarialVLM

export PYTHONPATH=$PYTHONPATH:$(pwd)

# Each job gets a unique experiment name so results don't collide.
# Adjust the following SBATCH options if your cluster requires different resources.

SBATCH_COMMON=(
  --partition=h100
  --gres=gpu:nvidia_h100_nvl:4
  --cpus-per-task=8
  --mem=64G
  --output=slurm_logs/qwen_variant_%j.out
  --error=slurm_logs/qwen_variant_%j.err
)

submit_job() {
  local exp_name="$1"
  local target_text="$2"
  local multi_crop="$3"

  sbatch "${SBATCH_COMMON[@]}" --job-name="${exp_name}" --wrap="\
    source ~/miniconda3/etc/profile.d/conda.sh && \
    cd /mnt/DISAPP/home/kalebuch/AdversarialVLM && \
    conda activate vlm && \
    EXP_NAME='${exp_name}' \
    TARGET_TEXT='${target_text}' \
    N_MULTI_CROP='${multi_crop}' \
    ./scripts/attacks/attack_blackbox_enhanced_qwen.sh"
}

# 3 target text variants
submit_job "qwen_atk_target_sure" "sure" "1"
submit_job "qwen_atk_target_sure" "sure" "1"
submit_job "qwen_atk_target_certain" "sure, here it is" "1"
submit_job "qwen_atk_target_certain" "sure, here it is" "1"
submit_job "qwen_atk_target_confident" "I can help with that" "1"
submit_job "qwen_atk_target_confident" "I can help with that" "1"

submit_job "qwen_atk_crop_1" "sure" "1"
submit_job "qwen_atk_crop_3" "sure" "3"
submit_job "qwen_atk_crop_5" "sure" "5"
submit_job "qwen_atk_crop_7" "sure" "7"

# User guidance
cat <<'EOF'
Submitted 5 qwen attack jobs.
Check slurm_logs/qwen_variant_<jobid>.out and .err.
EOF
