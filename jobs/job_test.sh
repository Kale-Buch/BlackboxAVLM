#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --output=slurm_logs/gpu_test.out
#SBATCH --error=slurm_logs/gpu_test.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:nvidia_h100_nvl:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

cd /mnt/DISAPP/home/kalebuch/AdversarialVLM
export PYTHONPATH=$PYTHONPATH:$(pwd)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm

./scripts/attacks/test_pipeline.sh