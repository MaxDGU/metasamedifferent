#!/bin/bash
#SBATCH --job-name=vanilla_nat_train # Job name
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of tasks
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --mem=16G                    # Memory per node (adjust if needed for vanilla)
#SBATCH --time=08:00:00              # Walltime limit (HH:MM:SS) - 300 epochs might take a while
#SBATCH --gres=gpu:1                 # Number of GPUs per node
#SBATCH --partition=pli              # Partition
#SBATCH --account=nam                # Account
#SBATCH --output=slurm_logs/vanilla_nat_train_%A_%a.out # Standard output and error log
#SBATCH --array=0-14                 # Array job: 3 architectures * 5 seeds = 15 jobs (0-14)

# --- Environment Setup ---
echo "Setting up environment for vanilla training..."
source /usr/share/Modules/init/bash # Initialize environment modules
module purge                         # Remove any inherited modules
module load anaconda3/2023.9          # Load Anaconda (adjust version if needed)
source activate tensorflow           # Activate your conda environment (ensure it has PyTorch)
echo "Conda environment activated: $CONDA_DEFAULT_ENV"
echo "Python executable: $(which python)"

# --- Project Setup ---
PROJECT_DIR="/scratch/gpfs/mg7411/samedifferent" # CHANGE THIS TO YOUR PROJECT DIRECTORY ON DELLA
TRAIN_SCRIPT_PATH="${PROJECT_DIR}/naturalistic/train_vanilla.py"

# --- Set PYTHONPATH --- 
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
echo "PYTHONPATH set to: ${PYTHONPATH}"

# --- Data Source ---
# Updated to the confirmed path on Della
DATA_DIR_ABS="/scratch/gpfs/mg7411/data/naturalistic/N_16/trainsize_6400_1200-300-100"

# --- Output Base Directory for Vanilla Models ---
LOG_BASE_VANILLA_DIR="${PROJECT_DIR}/logs_naturalistic_vanilla"
mkdir -p "${LOG_BASE_VANILLA_DIR}"

# --- Parameters ---
SEEDS=(42 123 789 555 999)
ARCHITECTURES=("conv2lr" "conv4lr" "conv6lr") # Using 'lr' suffix for consistency

# Calculate seed and architecture for this array task
NUM_SEEDS=${#SEEDS[@]}
NUM_ARCHS=${#ARCHITECTURES[@]}

SEED_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_ARCHS))
ARCH_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_ARCHS))

CURRENT_SEED=${SEEDS[$SEED_INDEX]}
CURRENT_ARCH=${ARCHITECTURES[$ARCH_INDEX]}

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running Vanilla Training with Seed: ${CURRENT_SEED}, Architecture: ${CURRENT_ARCH}"

# Create a unique log directory for this specific run
RUN_OUTPUT_DIR="${LOG_BASE_VANILLA_DIR}/${CURRENT_ARCH}/seed_${CURRENT_SEED}"
mkdir -p "${RUN_OUTPUT_DIR}"
echo "Logging and saving model to: ${RUN_OUTPUT_DIR}"

# --- Training Hyperparameters (as discussed) ---
OPTIMIZER_TYPE="sgd"
LEARNING_RATE=1e-5 # Same as MAML inner LR
NUM_EPOCHS=300     # Calculated for data parity with MAML
BATCH_SIZE_TRAIN=32
NUM_WORKERS_DL=2   # Number of DataLoader workers
PATIENCE_EARLY_STOP=20 # Adding a reasonable patience for early stopping

# --- Run Training Script ---
echo "Starting vanilla training script..."
python "${TRAIN_SCRIPT_PATH}" \
    --architecture "${CURRENT_ARCH}" \
    --seed "${CURRENT_SEED}" \
    --data_dir "${DATA_DIR_ABS}" \
    --output_dir "${RUN_OUTPUT_DIR}" \
    --optimizer "${OPTIMIZER_TYPE}" \
    --lr "${LEARNING_RATE}" \
    --epochs "${NUM_EPOCHS}" \
    --batch_size "${BATCH_SIZE_TRAIN}" \
    --num_workers "${NUM_WORKERS_DL}" \
    --patience "${PATIENCE_EARLY_STOP}" \
    --device "cuda"
    # --momentum 0.9 # Default in script if SGD
    # --weight_decay 0.0 # Default in script

echo "Vanilla training script finished for Seed: ${CURRENT_SEED}, Arch: ${CURRENT_ARCH}"
date

# --- Example: How to submit this script ---
# sbatch run_naturalistic_vanilla_train_della.slurm 