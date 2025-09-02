#!/bin/bash
#SBATCH --job-name=vanilla_sgd_test_array
#SBATCH --output=logs/vanilla_sgd_test_array_%A_%a.out
#SBATCH --error=logs/vanilla_sgd_test_array_%A_%a.err
#SBATCH --array=0-5  # 2 seeds x 3 architectures = 6 jobs (0-5)
#SBATCH --time=4:00:00
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
echo "Starting vanilla SGD validation QUICK TEST array job"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "Working directory: $(pwd)"

# Check GPU availability
nvidia-smi

# Define seeds and architectures for QUICK TEST
SEEDS=(42 43)  # Only 2 seeds for quick test
ARCHITECTURES=(conv2 conv4 conv6)

# Calculate which seed and architecture to use
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 2))  # 2 seeds
ARCH_IDX=$((SLURM_ARRAY_TASK_ID / 2))

SEED=${SEEDS[$SEED_IDX]}
ARCH=${ARCHITECTURES[$ARCH_IDX]}

echo "QUICK TEST - Running validation for:"
echo "  Seed: $SEED"
echo "  Architecture: $ARCH"
echo "  Task ID: $SLURM_ARRAY_TASK_ID (seed_idx=$SEED_IDX, arch_idx=$ARCH_IDX)"

# Run validation for this specific seed and architecture - QUICK TEST PARAMETERS
python scripts/validate_vanilla_sgd_architectures_fixed.py \
    --data_dir data/meta_h5/pb \
    --save_dir results/vanilla_sgd_validation_test_array \
    --epochs 5 \
    --batch_size 32 \
    --lr 1e-4 \
    --patience 3 \
    --seed $SEED \
    --architectures $ARCH

echo "QUICK TEST validation completed for seed $SEED, architecture $ARCH"
echo "Results saved to: results/vanilla_sgd_validation_test_array/seed_$SEED"
