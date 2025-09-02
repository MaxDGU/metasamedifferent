#!/bin/bash
#SBATCH --job-name=test_nat_vanilla
#SBATCH --output=logs/nat_test_vanilla_%j.log
#SBATCH --error=logs/nat_test_vanilla_%j.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --partition=pli
#SBATCH --account=nam

# Create logs directory
mkdir -p logs

# Load required modules
module load anaconda3
module load cuda/11.7.0


# Run the testing script
python test_naturalistic_vanilla.py \
    --model_dir /scratch/gpfs/mg7411/results/pb_baselines \
    --data_dir /scratch/gpfs/mg7411/data/naturalistic/N_16/trainsize_6400_1200-300-100 \
    --output_dir /scratch/gpfs/mg7411/results/naturalistic_test/vanilla \
    --batch_size 8 