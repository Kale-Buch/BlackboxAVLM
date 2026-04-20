#!/bin/bash

# Script for SafeBench testing
# Usage: ./scripts/evaluation/safebench_test.sh EXPERIMENT_NAME ITERATION MODEL_SUFFIX CUDA_NUM
# Set experiment parameters
EXP_NAME=$1
ITERATION=$2
MODEL_SUF=$3
CUDA_NUM=$4

echo "Running SafeBench test for experiment: $EXP_NAME, iteration: $ITERATION, model: $MODEL_SUF"

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$CUDA_NUM

# Run SafeBench universal test
python src/evaluation/SafeBench_universal.py \
    --exp "$EXP_NAME" \
    --iter "$ITERATION" \
    --model_suf "$MODEL_SUF" \
    --cuda_num "$CUDA_NUM"

echo "SafeBench testing completed. Results saved in tests/${EXP_NAME}_${ITERATION}/${MODEL_SUF}/" 
