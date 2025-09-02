#!/bin/bash
#SBATCH --job-name=aggregate_vanilla_sgd
#SBATCH --output=logs/aggregate_vanilla_sgd_%j.out
#SBATCH --error=logs/aggregate_vanilla_sgd_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --mem=8GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mg7411@princeton.edu

# Environment setup
echo "Starting vanilla SGD results aggregation"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Working directory: $(pwd)"

# Load modules
module load anaconda3/2023.9
conda activate metasamedifferent

# Run aggregation
echo "Aggregating results across 5 seeds for 3 architectures..."
python scripts/aggregate_vanilla_sgd_results.py \
    --results_dir results/vanilla_sgd_validation_array \
    --seeds 42 43 44 45 46 \
    --architectures CONV2 CONV4 CONV6 \
    --save_dir results/vanilla_sgd_validation_array

echo "Aggregation completed!"
echo "Check results in: results/vanilla_sgd_validation_array/"
