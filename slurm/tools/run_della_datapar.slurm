#!/bin/bash
#SBATCH --job-name=ideal_datapar
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4      # Number of CPUs
#SBATCH --mem=16G
#SBATCH --time=08:00:00          # Adjust as needed, e.g., 8 hours for the whole experiment
#SBATCH --gres=gpu:1             # Request 1 GPU (both single-task and MAML might use it)
#SBATCH --partition=pli          # Or your default partition
#SBATCH --account=nam            # Or your default account
#SBATCH --output=slurm_logs/ideal_datapar/job_%A.out # Path relative to submission dir
#SBATCH --error=slurm_logs/ideal_datapar/job_%A.err  # Path relative to submission dir

# --- User Configuration ---
NETID="mg7411" # Make sure this is correct

# --- Directory Setup ---
PROJECT_ROOT_ON_CLUSTER="/scratch/gpfs/${NETID}/samedifferent"
EXPERIMENT_SCRIPT_NAME="della_datapar.sh" # The script we want to run

# Absolute path to the experiment script
EXPERIMENT_SCRIPT_FULL_PATH="${PROJECT_ROOT_ON_CLUSTER}/${EXPERIMENT_SCRIPT_NAME}"

# Create Slurm log directory within the project structure if it doesn't exist
# (relative to where sbatch is called from, if CWD is project root)
SLURM_LOG_BASE_DIR="slurm_logs/ideal_datapar"
mkdir -p "${SLURM_LOG_BASE_DIR}" # This will be relative to CWD

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

# --- Run the Experiment Script ---
echo "Starting experiment script..."
echo "Job ID: $SLURM_JOB_ID"
echo "Running Experiment Script: ${EXPERIMENT_SCRIPT_FULL_PATH}"

# Change to the project root directory to ensure relative paths in della_datapar.sh work correctly
# and that slurm output logs go to the intended relative path if sbatch is run from here.
echo "Changing to project root: ${PROJECT_ROOT_ON_CLUSTER}"
cd "${PROJECT_ROOT_ON_CLUSTER}" || { echo "Failed to cd to ${PROJECT_ROOT_ON_CLUSTER}"; exit 1; }
echo "Current working directory: $(pwd)"

# Ensure the script to be run is executable
if [ -f "${EXPERIMENT_SCRIPT_NAME}" ]; then
    chmod +x "${EXPERIMENT_SCRIPT_NAME}"
else
    echo "Error: Experiment script ${EXPERIMENT_SCRIPT_NAME} not found in $(pwd)"
    exit 1
fi

echo "Running command: ./${EXPERIMENT_SCRIPT_NAME}"

./${EXPERIMENT_SCRIPT_NAME}

EXIT_CODE=$?
echo "Experiment script finished with exit code $EXIT_CODE."

conda deactivate
echo "Job finished at: $(date)" 