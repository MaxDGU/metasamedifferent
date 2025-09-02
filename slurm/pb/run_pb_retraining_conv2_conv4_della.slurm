#!/bin/bash
#SBATCH --job-name=pb-retrain-conv2-4
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --output=logs/pb_retrain_conv24_%A_%a.out

# --- Environment Setup ---
echo "Setting up the environment..."
source /usr/share/Modules/init/bash
module load anaconda3/2023.9
conda activate tensorflow
echo "Conda environment activated."

# --- Array Job Setup ---
ARCHS=(conv2 conv4)
SEEDS=(42 123 555 789 999)

# Calculate architecture and seed for this job
ARCH_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 5))

ARCH=${ARCHS[$ARCH_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

echo "Running job for ARCH=${ARCH} and SEED=${SEED}"

# --- Set Project Path ---
export GIT_REPO_PATH="/scratch/gpfs/mg7411/samedifferent"
export PYTHONPATH="${PYTHONPATH}:${GIT_REPO_PATH}"
cd "${GIT_REPO_PATH}"
echo "Current working directory: $(pwd)"
echo "PYTHONPATH: ${PYTHONPATH}"

# --- Execution ---
echo "Starting PB re-training with ${ARCH}lr for seed ${SEED}..."
python -m meta_baseline.train_and_test_meta_baselines \
    --data_dir /scratch/gpfs/mg7411/data/pb/pb \
    --output_dir /scratch/gpfs/mg7411/samedifferent/results/pb_retrained_${ARCH}lr \
    --architecture ${ARCH} \
    --seed ${SEED} \
    --epochs 100 \
    --batch_size 8 \
    --inner_lr 0.001 \
    --outer_lr 0.0001 \
    --support_size 4 6 8 10 \
    --test_support_size 10 \
    --adaptation_steps 5 \
    --test_adaptation_steps 15 \
    --patience 30 \
    --min_delta 0.005

echo "Job finished with exit code $? at $(date)" 