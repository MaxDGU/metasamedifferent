#!/bin/bash
#SBATCH --job-name=nat_test
#SBATCH --output=logs/nat_test_%j.log
#SBATCH --error=logs/nat_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4

# Load required modules
module purge
module load anaconda3/2023.3
eval "$(conda shell.bash hook)"
conda activate tensorflow

# Set environment variables for better CUDA memory management
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

# Create output directories
mkdir -p logs
mkdir -p results/naturalistic/conv2/seed_42

# Run a single test job with conservative settings
python train_vanilla_cluster.py \
    --architecture conv2 \
    --seed 42 \
    --data_dir data/naturalistic/N_16/trainsize_6400_1200-300-100 \
    --output_dir results/naturalistic/conv2/seed_42 \