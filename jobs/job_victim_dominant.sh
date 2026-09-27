#!/bin/bash
#SBATCH --job-name=victim_dom
#SBATCH --output=slurm_logs/victim_dom_%j.out
#SBATCH --error=slurm_logs/victim_dom_%j.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:nvidia_h100_nvl:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00

cd /mnt/DISAPP/home/kalebuch/AdversarialVLM
export PYTHONPATH=$PYTHONPATH:$(pwd)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm

mkdir -p slurm_logs runs/victim_dominant

# GPU layout (2-card only):
#   cuda:0 → Phi-3.5 + LLaVA-1.5
#   cuda:1 → LLaMA-3.2-11B + Qwen2-VL victim

echo "========================================================"
echo "SOLUTION 1: Victim-dominant hybrid test"
echo "Formula: g_victim + λ_surr * g_ens"
echo "Task: Is this image colorful? → Yes"
echo "========================================================"

for LAM in 0.0 0.1 0.5; do
    echo ""
    echo "--- Surrogate λ = ${LAM} ---"
    python scratch/run_victim_dominant.py \
        --image       images/gray.png  \
        --num_iter    100              \
        --step_size   0.01             \
        --epsilon     0.05             \
        --surr_lam    ${LAM}           \
        --prgf_lam    0.5              \
        --num_dirs    30               \
        --patch_size  32               \
        --eps_ball    0.05             \
        --eval_interval 10             \
        --surr0_dev   cuda:0           \
        --surr1_dev   cuda:1           \
        --surr2_dev   cuda:0           \
        --victim_dev  cuda:1           \
        --out_dir     runs/victim_dominant \
        --seed        42
done

echo ""
echo "========================================================"
echo "ALL DONE"
echo "========================================================"
