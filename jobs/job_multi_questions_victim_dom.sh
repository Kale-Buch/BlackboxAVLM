#!/bin/bash
#SBATCH --job-name=vdom_multiq
#SBATCH --output=slurm_logs/vdom_multiq_%j.out
#SBATCH --error=slurm_logs/vdom_multiq_%j.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:nvidia_h100_nvl:2
#SBATCH --exclude=rpg-93-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00

cd /mnt/DISAPP/home/kalebuch/AdversarialVLM
export PYTHONPATH=$PYTHONPATH:$(pwd)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm

mkdir -p slurm_logs runs/multi_question_victim_dom

echo "========================================================"
echo "MULTI-QUESTION VICTIM-DOMINANT RUN"
echo "Structured P-RGF (Patch-32, lambda=0.1, eps_ball=0.3)"
echo "Victim: Qwen2-VL-2B-Instruct"
echo "========================================================"

# Test across representative benchmark question indices from src/questions.py
QUESTION_INDICES=(0 1 2 3 4 5)

for Q_IDX in "${QUESTION_INDICES[@]}"; do
    echo ""
    echo "========================================================"
    echo ">>> Running Question Index ${Q_IDX} (200 iters) <<<"
    echo "========================================================"
    python src/victim_dom_test.py \
        --image          images/gray.png              \
        --question_idx   ${Q_IDX}                     \
        --target_text    "Sure"                       \
        --num_iter       2000                          \
        --step_size      0.02                         \
        --epsilon        0.05                         \
        --eps_ball       0.30                         \
        --patch_size     32                           \
        --eval_interval  25                           \
        --surr_lam       0.1                          \
        --prgf_lam       0.5                          \
        --num_dirs       100                          \
        --surr0_dev      cuda:0                       \
        --surr1_dev      cuda:1                       \
        --surr2_dev      cuda:0                       \
        --victim_dev     cuda:1                       \
        --victim_model   Qwen/Qwen2-VL-2B-Instruct    \
        --out_dir        runs/multi_question_victim_dom \
        --seed           42
done

echo ""
echo "========================================================"
echo "ALL MULTI-QUESTION RUNS COMPLETED"
echo "========================================================"
