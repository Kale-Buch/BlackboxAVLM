#!/bin/bash
#SBATCH --job-name=vdom_neutral
#SBATCH --output=slurm_logs/vdom_neutral_%j.out
#SBATCH --error=slurm_logs/vdom_neutral_%j.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:nvidia_h100_nvl:2
#SBATCH --exclude=rpg-93-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

cd /mnt/DISAPP/home/kalebuch/AdversarialVLM
export PYTHONPATH=$PYTHONPATH:$(pwd)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm

mkdir -p slurm_logs runs/multi_question_neutral

echo "========================================================"
echo "MULTI-QUESTION NEUTRAL / BALANCED CO-OPTIMIZATION RUN"
echo "Structured P-RGF with Balanced Gradient Scales"
echo "Victim: Qwen2-VL-2B-Instruct | Surrogates: Phi-3.5, LLaMA-3.2, LLaVA-1.5"
echo "========================================================"

# Test across representative benchmark question indices from src/questions.py
QUESTION_INDICES=(0 1 2 3 4 5)

for Q_IDX in "${QUESTION_INDICES[@]}"; do
    echo ""
    echo "========================================================"
    echo ">>> Running Neutral Multi-Question Index ${Q_IDX} <<<"
    echo "========================================================"
    python src/victim_dom_test.py \
        --image               images/gray.png              \
        --question_idx        ${Q_IDX}                     \
        --target_text         "Sure"                       \
        --num_iter            2000                         \
        --step_size           0.02                         \
        --epsilon             0.05                         \
        --eps_ball            0.30                         \
        --patch_size          32                           \
        --eval_interval       25                           \
        --balance_grad_scale                               \
        --surr_lam            1.0                          \
        --prgf_lam            0.5                          \
        --num_dirs            100                          \
        --surr0_dev           cuda:0                       \
        --surr1_dev           cuda:1                       \
        --surr2_dev           cuda:0                       \
        --victim_dev          cuda:1                       \
        --victim_model        Qwen/Qwen2-VL-2B-Instruct    \
        --out_dir             runs/multi_question_neutral  \
        --seed                42
done

echo ""
echo "========================================================"
echo "ALL NEUTRAL MULTI-QUESTION RUNS COMPLETED"
echo "========================================================"
