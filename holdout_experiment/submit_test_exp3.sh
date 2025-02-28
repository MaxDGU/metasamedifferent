#!/bin/bash
#SBATCH --job-name=test_exp3
#SBATCH --output=slurm_logs/test_exp3_%j.out
#SBATCH --error=slurm_logs/test_exp3_%j.err
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH --partition=pli
#SBATCH --account=nam

# Create logs directory
mkdir -p slurm_logs

# Load required modules
module load anaconda3
module load cuda/11.7.0

# Activate conda environment
source activate torch_env

# Run the testing script
python -m meta_baseline.test_experiment3 \
    --exp_dir exp3_holdout_runs_20250131_155132 \
    --data_dir data/pb/pb \
    --output_dir results/experiment3/test_results \
    --test_adaptation_steps 15 \
    --batch_size 16 