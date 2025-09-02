#!/bin/bash
#SBATCH --job-name=AdaptPCA      # Job name
#SBATCH --nodes=1                # Run on a single node
#SBATCH --ntasks=1               # Run a single task
#SBATCH --cpus-per-task=4        # Number of CPU cores per task
#SBATCH --mem=64G                # Job memory request (increased to 64G)
#SBATCH --time=01:00:00          # Time limit hrs:min:sec
#SBATCH --output=logs/adapt_pca_%j.out # Standard output and error log

# --- Environment Setup ---
echo "Purging modules"
module purge
echo "Loading anaconda..."
module load anaconda3/2023.9

# --- Activate Conda Environment ---
echo "Activating conda environment: tensorflow"
conda activate tensorflow

# --- Set Python Path ---
# Assuming you run 'sbatch' from the root of your 'metasamedifferent' repo
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"
echo "PYTHONPATH set to: $PYTHONPATH"

# --- Run the Python Script ---
echo "Running the adaptation and PCA script..."
# Using -u for unbuffered output, which is good for seeing logs in real-time
python -u scripts/adapt_and_visualize_pca.py

echo "Job finished." 