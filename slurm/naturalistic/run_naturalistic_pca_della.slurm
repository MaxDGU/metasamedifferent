#!/bin/bash
#SBATCH --job-name=nat_pca_viz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/naturalistic_pca_viz_%j.out

# Activate conda environment
source ~/.bashrc
conda activate tensorflow

# Navigate to the project directory
cd /scratch/gpfs/mg7411/samedifferent

echo "Setting up the environment..."
conda activate tensorflow
echo "Conda environment activated."
echo "Current working directory: $(pwd)"

# Set PYTHONPATH to include the project root
export PYTHONPATH="/scratch/gpfs/mg7411/samedifferent:$PYTHONPATH"
echo "PYTHONPATH set to: $PYTHONPATH"

SAVE_DIR="./visualizations/domain_adaptation_pca/naturalistic_static"

echo "Running Static PCA Visualization for Naturalistic Models"

python scripts/visualize_naturalistic_to_pb_adaptation.py --save_dir ${SAVE_DIR}

echo "Job finished with exit code $? at $(date)" 