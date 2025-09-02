#!/bin/bash
#SBATCH --job-name=WeightsTSNE     # Job name
#SBATCH --nodes=1                # Run on a single node
#SBATCH --ntasks=1               # Run a single task
#SBATCH --cpus-per-task=4        # Number of CPU cores per task
#SBATCH --mem=32G                # Job memory request
#SBATCH --time=01:00:00          # Time limit hrs:min:sec
#SBATCH --output=logs/tsne_%j.out # Standard output and error log

# --- Environment Setup ---
echo "Purging modules"
module purge
echo "Loading anaconda..."
module load anaconda3/2023.9

# --- Activate Conda Environment ---
echo "Activating conda environment: tensorflow"
conda activate tensorflow

# --- Set Python Path ---
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"
echo "PYTHONPATH set to: $PYTHONPATH"

# --- Run the Python Script ---
echo "Running the t-SNE visualization script..."
python -u scripts/visualize_tsne_weights.py

echo "Job finished." 