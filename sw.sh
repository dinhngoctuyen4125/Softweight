#!/bin/bash

#SBATCH --job-name=softweight
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --partition=defq
#SBATCH --qos=short
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

export CUDA_VISIBLE_DEVICES=0

python sw.py \
    --model_name_or_path "tummitum/codebert-deprecated" \
    --data_path "./data/codellama/D_forget.json" \
    --ckpt_dir "./ckpt" \
    --seed 2026