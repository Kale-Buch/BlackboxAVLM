#!/bin/bash
#SBATCH --job-name=long_vdom
#SBATCH --output=slurm_logs/long_vdom_%j.out
#SBATCH --error=slurm_logs/long_vdom_%j.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:nvidia_h100_nvl:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00

cd /mnt/DISAPP/home/kalebuch/AdversarialVLM
export PYTHONPATH=$PYTHONPATH:$(pwd)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm

mkdir -p slurm_logs runs/long_victim_dom

echo "========================================================"
echo "EXTENDED VICTIM-DOMINANT RUN (500 ITERATIONS)"
echo "Fixed Structured ZO Query Estimator (Patch-32)"
echo "Victim: Qwen2-VL-2B-Instruct"
echo "========================================================"

for LAM in 0.0 0.1; do
    echo ""
    echo "========================================================"
    echo ">>> Running Surrogate Lambda = ${LAM} (500 iters) <<<"
    echo "========================================================"
    python src/run_victim_dominant.py \
        --image          images/gray.png              \
        --num_iter       2000                          \
        --step_size      0.01                         \
        --epsilon        0.05                         \
        --eps_ball       0.05                         \
        --patch_size     32                           \
        --eval_interval  25                           \
        --surr_lam       ${LAM}                       \
        --prgf_lam       0.5                          \
        --num_dirs       30                           \
        --surr0_dev      cuda:0                       \
        --surr1_dev      cuda:1                       \
        --surr2_dev      cuda:0                       \
        --victim_dev     cuda:1                       \
        --victim_model   Qwen/Qwen2-VL-2B-Instruct    \
        --out_dir        runs/long_victim_dom         \
        --seed           42
done

echo ""
echo "========================================================"
echo "ALL LONG RUNS COMPLETED"
echo "========================================================"
