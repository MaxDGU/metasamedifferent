#!/bin/bash
#SBATCH --job-name=vanilla_dropout_trial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/dropout_trial_%j.out
#SBATCH --error=logs/dropout_trial_%j.err

# Activate conda environment
source ~/.bashrc
conda activate tensorflow

# Change to the correct project directory
cd /scratch/gpfs/mg7411/samedifferent/

echo "Starting dropout trial run..."
echo "Seed: 999 (trial seed)"
echo "Architecture: conv6"
echo "Epochs: 20"
echo "Dropout Rate: 0.3"

# Run the training script with custom dropout
python baselines/train_pb_models.py \
    --architecture conv6 \
    --test_task regular \
    --seed 999 \
    --epochs 20 \
    --dropout_rate_fc 0.3 \
    --output_dir /scratch/gpfs/mg7411/samedifferent/results/dropout_trial

echo "Trial run finished with exit code $?" 