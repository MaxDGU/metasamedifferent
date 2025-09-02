#!/bin/bash
#SBATCH --job-name=vanilla_nat_test # Job name for vanilla testing
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks
#SBATCH --cpus-per-task=4           # Number of CPU cores per task
#SBATCH --mem=16G                   # Memory per node
#SBATCH --time=01:00:00             # Walltime limit (HH:MM:SS) - Testing should be faster
#SBATCH --gres=gpu:1                # Number of GPUs per node (testing might still benefit from GPU)
#SBATCH --partition=pli             # Partition
#SBATCH --account=nam               # Account
#SBATCH --output=slurm_logs/vanilla_nat_test_%j.out # Standard output and error log

# --- Environment Setup ---
echo "Setting up environment for vanilla model testing..."
source /usr/share/Modules/init/bash # Initialize environment modules
module purge                        # Remove any inherited modules
module load anaconda3/2023.9         # Load Anaconda (adjust version if needed)
source activate tensorflow          # Activate your conda environment (ensure it has PyTorch, matplotlib)
echo "Conda environment activated: $CONDA_DEFAULT_ENV"
echo "Python executable: $(which python)"

# --- Project Setup ---
PROJECT_DIR="/scratch/gpfs/mg7411/samedifferent" # CHANGE THIS TO YOUR PROJECT DIRECTORY ON DELLA
TEST_SCRIPT_PATH="${PROJECT_DIR}/naturalistic/test_vanilla_model.py"

# --- Set PYTHONPATH to include the project root ---
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
echo "PYTHONPATH set to: ${PYTHONPATH}"

# --- Data Source for Testing ---
DATA_DIR_ABS="/scratch/gpfs/mg7411/data/naturalistic/N_16/trainsize_6400_1200-300-100"

# --- Trained Model Directory ---
# Base directory where trained vanilla models were saved by train_vanilla.py
BASE_LOG_DIR_VANILLA="${PROJECT_DIR}/logs_naturalistic_vanilla"

# --- Output Setup for Test Results ---
TEST_OUTPUT_DIR_VANILLA="${PROJECT_DIR}/test_results_vanilla" # Specific output dir for vanilla test results
mkdir -p "${TEST_OUTPUT_DIR_VANILLA}"
JSON_OUTPUT_PATH_VANILLA="${TEST_OUTPUT_DIR_VANILLA}/vanilla_test_summary.json"
PLOT_OUTPUT_PATH_VANILLA="${TEST_OUTPUT_DIR_VANILLA}/vanilla_test_accuracies.png"
echo "Vanilla test results (JSON/plot) will be saved to: ${TEST_OUTPUT_DIR_VANILLA}"

# --- Run Vanilla Testing Script ---
echo "Starting vanilla model testing script..."
python "${TEST_SCRIPT_PATH}" \
    --data_dir "${DATA_DIR_ABS}" \
    --base_log_dir "${BASE_LOG_DIR_VANILLA}" \
    --output_json_path "${JSON_OUTPUT_PATH_VANILLA}" \
    --output_plot_path "${PLOT_OUTPUT_PATH_VANILLA}" \
    --batch_size 32 \
    --num_workers 2 \
    --device "cuda"

echo "Vanilla model testing script finished."
date

# --- Example: How to submit this script ---
# sbatch run_naturalistic_vanilla_test_della.slurm 