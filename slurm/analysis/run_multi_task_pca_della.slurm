#!/bin/bash
#SBATCH --job-name=multi-task-pca     # Job name
#SBATCH --output=logs/multi_task_pca_%j.out # Standard output and error log
#SBATCH --nodes=1                     # Run all processes on a single node
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --mem=64G                     # Job memory request
#SBATCH --gres=gpu:1                  # Request one GPU
#SBATCH --time=02:00:00               # Time limit hrs:min:sec

# --- Environment Setup ---
echo "Setting up the environment..."
source /usr/share/Modules/init/bash
module load anaconda3/2023.9
conda activate tensorflow
echo "Conda environment activated."

# --- Set Project Path ---
export GIT_REPO_PATH="/scratch/gpfs/mg7411/samedifferent"
export PYTHONPATH="${PYTHONPATH}:${GIT_REPO_PATH}"
cd "${GIT_REPO_PATH}"
echo "Current working directory: $(pwd)"
echo "PYTHONPATH: ${PYTHONPATH}"

# --- Execution ---
echo "Running the Multi-Task Adaptation PCA script..."
python scripts/visualize_multi_task_adaptation_pca.py \
    --vanilla_seed 0 \
    --meta_seed 3 \
    --support_size 6 \
    --lr 0.001 \
    --steps 5

echo "Job finished with exit code $? at $(date)" 