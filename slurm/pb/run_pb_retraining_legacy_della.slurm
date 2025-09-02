#!/bin/bash
#SBATCH --job-name=pb-retrain-conv6
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/pb_retrain_conv6_%A_%a.out

# --- Environment Setup ---
echo "Setting up the environment..."
source /usr/share/Modules/init/bash
module load anaconda3/2023.9
conda activate tensorflow
echo "Conda environment activated."

# --- Array Job Setup ---
SEEDS=(42 123 555 789 999)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo "Running job for SEED=${SEED}"

# --- Set Project Path ---
export GIT_REPO_PATH="/scratch/gpfs/mg7411/samedifferent"
export PYTHONPATH="${PYTHONPATH}:${GIT_REPO_PATH}"
cd "${GIT_REPO_PATH}"
echo "Current working directory: $(pwd)"
echo "PYTHONPATH: ${PYTHONPATH}"

# --- Execution ---
# We are re-training the PB dataset using the LEGACY 6-layer CNN from baselines.
# This ensures the resulting model architecture is identical to the one used
# for the naturalistic experiments, allowing for a direct comparison.
echo "Starting PB re-training with LEGACY conv6 for seed ${SEED}..."
python -m meta_baseline.train_legacy_pb \
    --data_dir /scratch/gpfs/mg7411/samedifferent/data/meta_h5/pb \
    --output_dir /scratch/gpfs/mg7411/samedifferent/results/pb_retrained_legacy_conv6 \
    --architecture conv6 \
    --seed ${SEED} \
    --epochs 100 \
    --batch_size 8 \
    --inner_lr 0.001 \
    --outer_lr 0.0001 \
    --support_size 4 6 8 10 \
    --test_support_size 10 \
    --adaptation_steps 5 \
    --test_adaptation_steps 15

echo "Job finished with exit code $? at $(date)" 