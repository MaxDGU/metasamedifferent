#!/bin/bash
#SBATCH --job-name=ablation_conv2
#SBATCH --output=logs/ablation_conv2_%j.out
#SBATCH --error=logs/ablation_conv2_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1

# Load modules and activate environment
module purge
module load anaconda3/2023.9
conda activate tensorflow

# Change to project directory
cd /scratch/gpfs/mg7411/samedifferent

# Create logs directory if it doesn't exist
mkdir -p logs

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Run the ablation experiment
echo "Starting conv2 ablation experiment at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

python scripts/run_ablation_experiment.py

echo "Conv2 ablation experiment completed at $(date)" 