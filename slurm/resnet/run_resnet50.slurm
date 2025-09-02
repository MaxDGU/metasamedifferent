#!/bin/bash
#SBATCH --job-name=resnet18_sd        # Job name
#SBATCH --output=resnet18_%j.log      # Output file name (%j expands to jobID)
#SBATCH --error=resnet18_%j.err       # Error file name (%j expands to jobID)
#SBATCH --nodes=1                     # Run on a single node
#SBATCH --ntasks-per-node=1           # Run a single task
#SBATCH --cpus-per-task=5             # 4 CPU cores for data loading + 1 for main process
#SBATCH --mem=64G                     # More memory for ResNet model and data
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --time=48:00:00              # Time limit hrs:min:sec (increased for ResNet)
#SBATCH --mail-type=BEGIN,END,FAIL    # Mail events
#SBATCH --mail-user=mg7411@princeton.edu  # Where to send mail

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"

# Load necessary modules
module purge
module load anaconda/3

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate your environment
conda activate tensorflow

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Navigate to project directory
cd samedifferent/metasamedifferent

# Run the script
python resnet18.py

# Print completion time
echo "End time: $(date)" 