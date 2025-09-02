#!/bin/bash
#SBATCH --job-name=vanilla_pb_train_all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/vanilla_pb_train_all_%A_%a.out
#SBATCH --array=0-4

# Activate conda environment
source ~/.bashrc
conda activate tensorflow

# Navigate to the project directory
cd /scratch/gpfs/mg7411/samedifferent

echo "Setting up the environment..."
conda activate tensorflow
echo "Conda environment activated."
echo "Current working directory: $(pwd)"

# Set PYTHONPATH to include the project root
export PYTHONPATH="/scratch/gpfs/mg7411/samedifferent:$PYTHONPATH"
echo "PYTHONPATH set to: $PYTHONPATH"

# Define seeds
SEEDS=(123 42 555 789 999)

# Get the seed for the current job array index
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "Running training on ALL tasks for seed=${SEED}"

python baselines/train_vanilla_pb_final.py --seed ${SEED}

echo "Job finished with exit code $? at $(date)" 