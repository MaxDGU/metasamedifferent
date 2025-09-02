#!/bin/bash
#SBATCH --job-name=vanilla_sgd_validation
#SBATCH --output=logs/vanilla_sgd_validation_%j.out
#SBATCH --error=logs/vanilla_sgd_validation_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mg7411@princeton.edu

# Environment setup
echo "Starting vanilla SGD architecture validation job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "Working directory: $(pwd)"

# Load modules
module load anaconda3/2023.9
conda activate metasamedifferent

# Check GPU availability
nvidia-smi

# Run validation across all architectures
echo "Running vanilla SGD validation across conv2, conv4, and conv6 architectures..."
python scripts/validate_vanilla_sgd_architectures.py \
    --data_dir /scratch/gpfs/mg7411/samedifferent/data/vanilla_h5 \
    --save_dir /scratch/gpfs/mg7411/samedifferent/results/vanilla_sgd_validation \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --patience 10 \
    --seed 42 \
    --architectures conv2 conv4 conv6

echo "Validation completed!"
echo "Results saved to: /scratch/gpfs/mg7411/samedifferent/results/vanilla_sgd_validation"
