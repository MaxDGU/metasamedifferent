#!/bin/bash
#SBATCH --job-name=circuit_analysis
#SBATCH --output=slurm_logs/circuit_analysis_%A.out
#SBATCH --error=slurm_logs/circuit_analysis_%A.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4

# Create logs directory
mkdir -p slurm_logs

# Load required modules
module load anaconda3
module load cuda/11.7.0

# Activate conda environment
source activate tensorflow

echo "Running circuit analysis baseline training"

# Run the training script
python scripts/find_and_ablate_circuit.py \
    --mode train \
    --epochs 50 