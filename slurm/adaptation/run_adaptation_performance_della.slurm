#!/bin/bash
#SBATCH --job-name=adapt-perf-viz      # Job name
#SBATCH --output=logs/adapt_perf_%j.out # Standard output and error log
#SBATCH --nodes=1                      # Run all processes on a single node
#SBATCH --ntasks=1                     # Run a single task
#SBATCH --cpus-per-task=4              # Number of CPU cores per task
#SBATCH --mem=64G                      # Job memory request
#SBATCH --gres=gpu:1                   # Request one GPU
#SBATCH --time=02:00:00                # Time limit hrs:min:sec

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
echo "Running the Adaptation Performance Visualization script..."
python scripts/visualize_adaptation_performance.py \
    --vanilla_models_dir "/scratch/gpfs/mg7411/samedifferent/single_task/results/pb_single_task" \
    --meta_models_dir "/scratch/gpfs/mg7411/samedifferent/maml_pbweights_conv6" \
    --data_dir "/scratch/gpfs/mg7411/samedifferent/data/naturalistic" \
    --output_dir "/scratch/gpfs/mg7411/samedifferent/visualizations/adaptation_performance" \
    --lr 0.01 \
    --adaptation_steps 100 \
    --batch_size 64

echo "Job finished with exit code $? at $(date)" 