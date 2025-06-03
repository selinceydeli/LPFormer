#!/bin/bash

#SBATCH --job-name="LPFormer_pubmed"
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
cd /home/sceydeli/

# Activate your conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda init
conda activate miniconda3/envs/myenv
# conda activate myenv

# Verify environment is active
echo "Active conda environment: $CONDA_DEFAULT_ENV"
which python

# Change to LPFormer directory
cd LPFormer

# Create checkpoints directory if it doesn't exist
mkdir -p checkpoints/pubmed

# Run the LPFormer training
# For testing, reduce epochs and runs
python src/run.py \
    --data_name pubmed \
    --lr 1e-3 \
    --gnn-layers 1 \
    --dim 128 \
    --batch-size 1024 \
    --epochs 100 \
    --eps 1e-5 \
    --gnn-drop 0.3 \
    --dropout 0.3 \
    --pred-drop 0.3 \
    --att-drop 0.3 \
    --num-heads 1 \
    --thresh-1hop 1e-2 \
    --thresh-non1hop 1e-2 --mask-input \
    --feat-drop 0.3 \
    --l2 1e-4 \
    --eval_steps 1 \
    --decay 1 \
    --non-verbose \
    --runs 10 \
    --device 0 > pubmed_training.log 2>&1

echo "Training completed. Check pubmed_training.log for results."