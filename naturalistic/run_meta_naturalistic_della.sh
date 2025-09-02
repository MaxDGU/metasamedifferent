#!/bin/bash
#SBATCH --job-name=meta_nat_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4  # Number of CPUs, can be num_workers for DataLoader + 1
#SBATCH --mem=16G
#SBATCH --time=08:00:00      # Adjust as needed, e.g., 8 hours
#SBATCH --gres=gpu:1         # Request 1 GPU
#SBATCH --partition=pli
#SBATCH --account=nam
#SBATCH --output=slurm_logs/naturalistic/meta_train_%A_%a.out # Path relative to submission dir
#SBATCH --error=slurm_logs/naturalistic/meta_train_%A_%a.err  # Path relative to submission dir
#SBATCH --array=0-14 # 15 jobs total

# --- User Configuration ---
# Ensure NETID is set if used, though paths are now more explicit
NETID="mg7411" # !!! IMPORTANT: REPLACE YOUR_NETID if not mg7411 !!!

# --- Directory Setup ---
# PROJECT_ROOT_ON_CLUSTER is where the 'samedifferent' repo is cloned on Della
PROJECT_ROOT_ON_CLUSTER="/scratch/gpfs/${NETID}/samedifferent"
SCRIPT_SUBDIR="naturalistic" # Subdirectory containing the training script
PYTHON_SCRIPT_NAME="meta_naturalistic_train.py" # Name of the training script

# Absolute path to the training script
PYTHON_SCRIPT_FULL_PATH="${PROJECT_ROOT_ON_CLUSTER}/${SCRIPT_SUBDIR}/${PYTHON_SCRIPT_NAME}"

# Absolute path to the data directory
DATA_DIR_ON_CLUSTER="/scratch/gpfs/${NETID}/data/naturalistic/meta"

# Base output directory for results and logs, within the script's subdirectory structure
BASE_OUTPUT_DIR_ON_CLUSTER="${PROJECT_ROOT_ON_CLUSTER}/${SCRIPT_SUBDIR}/results_meta_della"
SLURM_LOG_DIR_IN_PROJECT="${PROJECT_ROOT_ON_CLUSTER}/${SCRIPT_SUBDIR}/slurm_logs/meta_della"

# Override SBATCH output/error paths to be absolute within the project structure
# This makes them independent of where sbatch is run from, provided PROJECT_ROOT_ON_CLUSTER is correct.
# Format: %A for Job ID, %a for Array Task ID
# Ensure these lines are BEFORE any command that might produce output (like module load)
# We'll create these directories right after defining them.
# shellcheck disable=SC2154 
# (SLURM_JOB_ID and SLURM_ARRAY_TASK_ID are set by Slurm)
# Note: Slurm might not expand these variables early enough for #SBATCH directives.
# So, the relative paths in #SBATCH are kept, and we ensure the CWD makes them work.
# For actual log files, it's better to ensure the CWD is the project root when submitting,
# or use absolute paths in #SBATCH directives if supported for your Slurm version for variables.

mkdir -p "${SLURM_LOG_DIR_IN_PROJECT}" # Ensure Slurm log directory exists

# --- Parameters for meta_naturalistic_train.py ---
EPOCHS=100
EPISODES_PER_EPOCH=500
META_LR=0.0001       # Updated default
INNER_LR=0.0001      # Updated default
INNER_STEPS=5
WEIGHT_DECAY=0.01
GRAD_CLIP_NORM=1.0   # New parameter, using new default
NUM_WORKERS=2
PATIENCE=15

# --- Array Job Setup ---
ARCHITECTURES=("conv2lr" "conv4lr" "conv6lr")
SEEDS=(0 1 2 3 4) # Using 5 seeds

NUM_ARCHS=${#ARCHITECTURES[@]}
NUM_SEEDS=${#SEEDS[@]}

TOTAL_JOBS=$((NUM_ARCHS * NUM_SEEDS))

# Calculate architecture and seed for the current Slurm array task ID
ARCH_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))

CURRENT_ARCH=${ARCHITECTURES[$ARCH_INDEX]}
CURRENT_SEED=${SEEDS[$SEED_INDEX]}

# --- Construct Output Directory for this specific job ---
# This will be passed as --log_dir to the python script
JOB_OUTPUT_DIR="${BASE_OUTPUT_DIR_ON_CLUSTER}/${CURRENT_ARCH}/seed_${CURRENT_SEED}"
mkdir -p "${JOB_OUTPUT_DIR}"

# --- Environment Setup ---
echo "Host: $(hostname)"
echo "Time: $(date)"
echo "Loading modules..."
module purge # Start with a clean environment
module load anaconda3/2023.9 # Or your preferred Anaconda module on Della
module load cudatoolkit/11.8      # Or your preferred CUDA module

# Activate your conda environment
CONDA_ENV_NAME="tensorflow" # Replace with your environment name
echo "Activating conda environment: ${CONDA_ENV_NAME}..."
# shellcheck source=/dev/null
source activate "${CONDA_ENV_NAME}" # Use 'source' not 'conda activate' in bash scripts

# --- Run the Python Training Script ---
echo "Starting Python script..."
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running Architecture: ${CURRENT_ARCH}, Seed: ${CURRENT_SEED}"
echo "Data Directory: ${DATA_DIR_ON_CLUSTER}"
echo "Outputting to: ${JOB_OUTPUT_DIR}"
echo "Python script path: ${PYTHON_SCRIPT_FULL_PATH}"

# Change to the project root directory to ensure relative paths in the script (if any) and for #SBATCH log paths work
cd "${PROJECT_ROOT_ON_CLUSTER}"
echo "Current working directory: $(pwd)"

echo "Running command:"
echo "python "${SCRIPT_SUBDIR}/${PYTHON_SCRIPT_NAME}" \
    --model "${CURRENT_ARCH}" \
    --seed "${CURRENT_SEED}" \
    --data_dir "${DATA_DIR_ON_CLUSTER}" \
    --log_dir "${JOB_OUTPUT_DIR}" \
    --epochs "${EPOCHS}" \
    --episodes_per_epoch "${EPISODES_PER_EPOCH}" \
    --meta_lr "${META_LR}" \
    --inner_lr "${INNER_LR}" \
    --inner_steps "${INNER_STEPS}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --grad_clip_norm "${GRAD_CLIP_NORM}" \
    --num_workers "${NUM_WORKERS}" \
    --patience "${PATIENCE}" \
    --device cuda"

python "${SCRIPT_SUBDIR}/${PYTHON_SCRIPT_NAME}" \
    --model "${CURRENT_ARCH}" \
    --seed "${CURRENT_SEED}" \
    --data_dir "${DATA_DIR_ON_CLUSTER}" \
    --log_dir "${JOB_OUTPUT_DIR}" \
    --epochs "${EPOCHS}" \
    --episodes_per_epoch "${EPISODES_PER_EPOCH}" \
    --meta_lr "${META_LR}" \
    --inner_lr "${INNER_LR}" \
    --inner_steps "${INNER_STEPS}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --grad_clip_norm "${GRAD_CLIP_NORM}" \
    --num_workers "${NUM_WORKERS}" \
    --patience "${PATIENCE}" \
    --device cuda

EXIT_CODE=$?
echo "Python script finished with exit code $EXIT_CODE for Arch: ${CURRENT_ARCH}, Seed: ${CURRENT_SEED}."

conda deactivate
echo "Job finished at: $(date)" 