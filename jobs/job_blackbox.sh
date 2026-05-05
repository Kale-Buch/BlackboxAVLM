#!/bin/bash
#SBATCH --job-name=gpu_new_blackbox
#SBATCH --output=slurm_logs/gpu_new_blackbox.out
#SBATCH --error=slurm_logs/gpu_new_blackbox.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:nvidia_h100_nvl:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G

cd /mnt/CS5379/home/kalebuch/AdversarialVLM
export PYTHONPATH=$PYTHONPATH:$(pwd)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm

./scripts/attacks/automated_pipeline.sh
