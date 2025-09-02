#!/bin/bash
#SBATCH --job-name=debug_pca_weights
#SBATCH --output=output_files/debug_pca_weights_%j.out
#SBATCH --error=output_files/debug_pca_weights_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load Python module
module purge
module load anaconda3/2023.9

# Activate conda environment
source activate /home/mg7411/.conda/envs/pytorch_env

# Change to project directory
cd /scratch/gpfs/mg7411/samedifferent

# Run the diagnostic script
echo "Running diagnostic PCA script..."
python scripts/debug_pca_weights.py

echo "Job completed at $(date)" 