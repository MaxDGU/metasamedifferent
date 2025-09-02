#!/bin/bash
#SBATCH --job-name=weight_space_viz
#SBATCH --output=/scratch/gpfs/mg7411/samedifferent/slurm/weight_viz_%A.out
#SBATCH --error=/scratch/gpfs/mg7411/samedifferent/slurm/weight_viz_%A.err
#SBATCH --time=01:00:00
#SBATCH --mem=128G
#SBATCH --partition=pli
#SBATCH --account=nam

# Load required modules
module purge
module load anaconda3/2023.9
# The script runs on CPU, so CUDA is not required
# module load cuda/11.8
conda activate tensorflow

# Ensure the visualizations directory exists
mkdir -p /scratch/gpfs/mg7411/samedifferent/visualizations

# Run the visualization script
# The script now takes a single output directory
python -u scripts/visualize_weight_space.py \
    --output_dir /scratch/gpfs/mg7411/samedifferent/visualizations

echo "Visualization job finished." 