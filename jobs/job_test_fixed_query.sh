#!/bin/bash
#SBATCH --job-name=test_fixed_query
#SBATCH --output=slurm_logs/test_fixed_query_%j.out
#SBATCH --error=slurm_logs/test_fixed_query_%j.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

cd /mnt/DISAPP/home/kalebuch/AdversarialVLM
export PYTHONPATH=$PYTHONPATH:$(pwd)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm

mkdir -p slurm_logs

python scratch/test_fixed_query.py
