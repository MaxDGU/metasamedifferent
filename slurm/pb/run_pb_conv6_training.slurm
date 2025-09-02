#!/bin/bash
#SBATCH --job-name=pb_conv6_training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/pb_conv6_%A_%a.out
#SBATCH --error=slurm_logs/pb_conv6_%A_%a.err
#SBATCH --array=0-4

# Define the seeds for the job array
SEEDS=(47 48 49 50 51)
CURRENT_SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Activate your conda environment
source /usr/licensed/anaconda3/2022.5/etc/profile.d/conda.sh
conda activate your_conda_env_name # IMPORTANT: Please replace with your actual conda environment name

# Run the training script
echo "Running PB Conv6 training for seed $CURRENT_SEED"
python baselines/train_pb_models.py \
    --architecture conv6 \
    --test_task regular \
    --seed $CURRENT_SEED \
    --output_dir /scratch/gpfs/mg7411/samedifferent/results/pb_baselines_vanilla_final

echo "Job finished for seed $CURRENT_SEED" 