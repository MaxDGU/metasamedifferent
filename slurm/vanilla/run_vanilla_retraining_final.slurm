#!/bin/bash
#SBATCH --job-name=vanilla_conv6_retrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/vanilla_conv6_retrain_%A_%a.out
#SBATCH --error=logs/vanilla_conv6_retrain_%A_%a.err
#SBATCH --array=0-4

# Define arrays for seeds
SEEDS=(47 48 49 50 51)

# Get the seed for the current array job
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Activate conda environment
source ~/.bashrc
conda activate tensorflow

# Change to the correct project directory
cd /scratch/gpfs/mg7411/samedifferent/

echo "Starting vanilla conv6 retraining..."
echo "Seed: $SEED"
echo "Architecture: conv6"
echo "Test Task: regular"

# Run the training script
python baselines/train_pb_models.py \
    --architecture conv6 \
    --test_task regular \
    --seed $SEED \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001 \
    --patience 10 \
    --output_dir /scratch/gpfs/mg7411/samedifferent/results/pb_baselines_vanilla_final

echo "Job finished with exit code $?" 