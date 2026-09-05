#!/bin/bash

#SBATCH --job-name=sw
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
    --dep_data "./data/codellama/D_test_U_dep.json" \
    --nondep_data "./data/codellama/D_test_U_nondep.json" \
    --ocsvm_repo "tummitum/OCSVM" \
    --seed 2026 \
    --num_test_samples 100 \
    --max_seq_length 512 \
    --batch_size 8 \
    --output_file "./weight_results.json"