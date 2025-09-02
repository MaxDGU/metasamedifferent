#!/bin/bash
#SBATCH --job-name=aggregate_vanilla_sgd
#SBATCH --output=logs/aggregate_vanilla_sgd_%j.out
#SBATCH --error=logs/aggregate_vanilla_sgd_%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB

# Load required modules (SAME AS WORKING SCRIPTS)
module load anaconda3/2022.5
module load cudatoolkit/12.6

# Activate conda environment (CORRECT ENVIRONMENT NAME)
source /usr/licensed/anaconda3/2022.5/etc/profile.d/conda.sh
conda activate tensorflow

# Create logs directory
mkdir -p logs

# Environment setup
echo "Starting vanilla SGD results aggregation"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Working directory: $(pwd)"

# Run aggregation
echo "Aggregating results across 5 seeds for 3 architectures..."
python scripts/aggregate_vanilla_sgd_results.py \
    --results_dir results/vanilla_sgd_validation_array \
    --seeds 42 43 44 45 46 \
    --architectures CONV2 CONV4 CONV6 \
    --save_dir results/vanilla_sgd_validation_array

echo "Aggregation completed!"
echo "Check results in: results/vanilla_sgd_validation_array/"
