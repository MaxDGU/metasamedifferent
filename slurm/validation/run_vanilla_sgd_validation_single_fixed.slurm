#!/bin/bash
#SBATCH --job-name=vanilla_sgd_validation_single
#SBATCH --output=logs/vanilla_sgd_validation_single_%j.out
#SBATCH --error=logs/vanilla_sgd_validation_single_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1

# Load required modules (SAME AS WORKING SCRIPTS)
module load anaconda3/2022.5
module load cudatoolkit/12.6

# Activate conda environment (CORRECT ENVIRONMENT NAME)
source /usr/licensed/anaconda3/2022.5/etc/profile.d/conda.sh
conda activate tensorflow

# Create logs directory
mkdir -p logs

# Environment setup
echo "Starting SINGLE vanilla SGD architecture validation job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "Working directory: $(pwd)"

# Check GPU availability
nvidia-smi

# Run validation across all architectures using CORRECT data loading
echo "Running vanilla SGD validation across conv2, conv4, and conv6 architectures..."
python scripts/validate_vanilla_sgd_architectures_fixed.py \
    --data_dir data/meta_h5/pb \
    --save_dir results/vanilla_sgd_validation_single \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --patience 10 \
    --seed 42 \
    --architectures conv2 conv4 conv6

echo "Validation completed!"
echo "Results saved to: results/vanilla_sgd_validation_single"
