#!/bin/bash
#SBATCH --job-name=meta_nat_test # Job name
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks=1               # Number of tasks
#SBATCH --cpus-per-task=4        # Number of CPU cores per task
#SBATCH --mem=16G                # Memory per node
#SBATCH --time=02:00:00          # Walltime limit (HH:MM:SS)
#SBATCH --gres=gpu:1             # Number of GPUs per node
#SBATCH --partition=pli          # Partition
#SBATCH --account=nam            # Account
#SBATCH --output=slurm_logs/meta_nat_test_%j.out # Standard output and error log

# --- Environment Setup ---
echo "Setting up environment..."
source /usr/share/Modules/init/bash # Initialize environment modules
module purge                     # Remove any inherited modules
module load anaconda3/2023.9      # Load Anaconda (adjust version if needed)
source activate tensorflow       # Activate your conda environment (ensure this has PyTorch, matplotlib, learn2learn)
echo "Conda environment activated: $CONDA_DEFAULT_ENV"
echo "Python executable: $(which python)"
echo "Running on host: $(hostname)"
echo "Running on GPU: $CUDA_VISIBLE_DEVICES"

# --- Project Setup ---
PROJECT_DIR="/scratch/gpfs/mg7411/samedifferent" # CHANGE THIS TO YOUR PROJECT DIRECTORY ON DELLA
TEST_SCRIPT_PATH="${PROJECT_DIR}/naturalistic/test_meta_model.py"
DATA_DIR_ABS="${PROJECT_DIR}/data/naturalistic"
BASE_LOG_DIR_ABS="${PROJECT_DIR}/logs_naturalistic_meta" # Directory where trained models are stored

# --- Output Setup ---
# Define a directory for test results (JSON, plot)
TEST_OUTPUT_DIR="${PROJECT_DIR}/test_results_meta"
mkdir -p "${TEST_OUTPUT_DIR}"
JSON_OUTPUT_PATH="${TEST_OUTPUT_DIR}/meta_test_summary.json"
PLOT_OUTPUT_PATH="${TEST_OUTPUT_DIR}/meta_test_accuracies.png"
echo "Test results (JSON/plot) will be saved to: ${TEST_OUTPUT_DIR}"

# --- Run Testing Script ---
echo "Starting meta-model testing script..."
python "${TEST_SCRIPT_PATH}" \
    --data_dir "${DATA_DIR_ABS}" \
    --base_log_dir "${BASE_LOG_DIR_ABS}" \
    --output_json_path "${JSON_OUTPUT_PATH}" \
    --output_plot_path "${PLOT_OUTPUT_PATH}" \
    --inner_lr 1e-5 \
    --inner_steps 5 \
    --device "cuda" \
    --num_workers 2 # Optional: adjust based on system and HDF5 reading performance
    # --max_test_episodes 10 # Optional: Uncomment for a quick debug run with fewer episodes

echo "Testing script finished."
date

# --- Example: How to submit this script ---
# sbatch run_naturalistic_meta_test_della.slurm
#
# To check job status:
# squeue -u $USER
#
# To view output:
# Check the slurm_logs directory specified in #SBATCH --output and contents of TEST_OUTPUT_DIR 