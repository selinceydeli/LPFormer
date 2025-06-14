#!/bin/bash

#SBATCH --job-name="LPFormer_citeseer"
#SBATCH --partition=gpu-a100
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=2
#SBATCH --mem-per-cpu=6GB

# Load modules
module load 2023r1
module load python
module load cuda

# Change to the home directory
cd /home/lemonhe/

# Activate your conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda init
conda activate miniconda3/envs/myenv
# conda activate myenv

# Verify environment is active
echo "Active conda environment: $CONDA_DEFAULT_ENV"
which python

# Print current timestamp
echo "Current timestamp: $(date)"

# Change to LPFormer directory
cd LPFormer

# Create checkpoints directory if it doesn't exist
mkdir -p checkpoints/citeseer

# Run the LPFormer training
# For testing, reduce epochs and runs
/home/lemonhe/miniconda3/envs/myenv/bin/python src/run.py \
    --data_name citeseer \
    --lr 5e-3 \
    --gnn-layers 1 \
    --dim 256 \
    --batch-size 1024 \
    --epochs 100 \
    --kill_cnt 100 \
    --eps 1e-7 \
    --gnn-drop 0.1 \
    --dropout 0.1 \
    --pred-drop 0.1 \
    --att-drop 0.1 \
    --num-heads 1 \
    --thresh-1hop 1e-2 \
    --thresh-non1hop 1 \
    --feat-drop 0.1 \
    --eval_steps 1 \
    --decay 0.95 \
    --non-verbose \
    --l2 0 \
    --runs 10 \
    --device 0 > citeseer_training_custom.log 2>&1

echo "Training completed. Check citeseer_training_custom.log for results."