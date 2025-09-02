#!/bin/bash
#SBATCH --job-name=test_nat_meta
#SBATCH --output=logs/nat_test_meta_%j.log
#SBATCH --error=logs/nat_test_meta_%j.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=18GB
#SBATCH --cpus-per-task=4
#SBATCH --partition=pli
#SBATCH --account=nam

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules
module load anaconda3/2023.3
module load cuda/11.8

# Activate conda environment
source activate tensorflow

# Run the testing script for each model and seed combination
for model in conv2 conv4; do
    for seed in 0 1 2 3 4 5 6 7 8 9; do
        echo "Testing $model with seed $seed"
        python test_naturalistic_meta.py \
            --model $model \
            --seed $seed \
            --data_dir /scratch/gpfs/mg7411/data/naturalistic/N_16/trainsize_6400_1200-300-100/test \
            --output_dir /scratch/gpfs/mg7411/same_different_paper/metasamedifferent/naturalistic/results
    done
done 