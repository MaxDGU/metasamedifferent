#!/bin/bash
#SBATCH --job-name=missing_pb
#SBATCH --output=missing_pb_%A_%a.out
#SBATCH --error=missing_pb_%A_%a.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=nam
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-299  # 10 tasks * 3 architectures * 10 seeds = 300 jobs

# Activate conda environment
source ~/.conda/etc/profile.d/conda.sh
conda activate tensorflow

# Set absolute path for output directory
OUTPUT_DIR="/scratch/gpfs/mg7411/results/pb_baselines"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Define arrays of tasks and architectures
tasks=("regular" "lines" "open" "wider_line" "scrambled" "random_color" "arrows" "irregular" "filled" "original")
architectures=("conv2" "conv4" "conv6")
seeds=(52 53 54 55 56 57 58 59 60 61)  # 10 new seeds

# Calculate which combination to run based on array index
n_archs=${#architectures[@]}
n_seeds=${#seeds[@]}

task_idx=$(( SLURM_ARRAY_TASK_ID / (n_archs * n_seeds) ))
arch_idx=$(( (SLURM_ARRAY_TASK_ID % (n_archs * n_seeds)) / n_seeds ))
seed_idx=$(( SLURM_ARRAY_TASK_ID % n_seeds ))

task=${tasks[$task_idx]}
arch=${architectures[$arch_idx]}
seed=${seeds[$seed_idx]}

echo "Running combination $SLURM_ARRAY_TASK_ID: task=$task, architecture=$arch, seed=$seed"
echo "Results will be saved to: $OUTPUT_DIR"

# Run the specific combination
python train_pb_models.py \
    --task "$task" \
    --architecture "$arch" \
    --seed "$seed" \
    --output_dir "$OUTPUT_DIR"