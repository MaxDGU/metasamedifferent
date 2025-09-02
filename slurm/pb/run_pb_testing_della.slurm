#!/bin/bash
#SBATCH --job-name=pb-test-conv6
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/pb_test_conv6_%A_%a.out

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
echo "Starting PB testing with conv6lr for seed ${SEED}..."
python -m meta_baseline.test_meta_baselines \
    --data_dir /scratch/gpfs/mg7411/data/pb/pb \
    --results_dir /scratch/gpfs/mg7411/samedifferent/results/pb_retrained_conv6lr \
    --architecture conv6 \
    --seed ${SEED} \
    --test_support_size 10 \
    --test_adaptation_steps 15

echo "Job finished with exit code $? at $(date)" 