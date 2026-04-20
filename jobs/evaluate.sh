#!/bin/bash
#SBATCH --job-name=find_it_job
#SBATCH --output=slurm_logs/find_it_job.out
#SBATCH --error=slurm_logs/find_it_job.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:nvidia_h100_nvl:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
cd /mnt/CS5379/home/kalebuch/AdversarialVLM
export PYTHONPATH=$PYTHONPATH:$(pwd)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlmg

./scripts/evaluation/find_best_iter.sh 
