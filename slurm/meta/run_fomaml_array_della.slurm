#!/bin/bash
#SBATCH --job-name=fomaml_array     # Job name
#SBATCH --output=logs/fomaml_array_%A_%a.out  # Standard output and error log
#SBATCH --error=logs/fomaml_array_%A_%a.err
#SBATCH --nodes=1                   # Run all processes on a single node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=4           # Number of CPU cores per task
#SBATCH --mem=16GB                  # Job memory request
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --time=8:00:00             # Time limit hrs:min:sec
#SBATCH --array=0-14                # Array range: 3 architectures * 5 seeds = 15 jobs

# Define the arrays for architectures and seeds
ARCHS=('conv2' 'conv4' 'conv6')
SEEDS=(42 43 44 45 46)

# Calculate the architecture and seed index from the SLURM array task ID
NUM_SEEDS=${#SEEDS[@]}
ARCH_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))

# Get the specific architecture and seed for this job
ARCH=${ARCHS[$ARCH_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

echo "Running FOMAML Experiment"
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Architecture: $ARCH"
echo "Seed: $SEED"

# Load conda environment
# IMPORTANT: Replace 'tensorflow' with the name of your conda environment if different
source /usr/licensed/anaconda3/2022.5/etc/profile.d/conda.sh
conda activate tensorflow

# Define project directory and output directory
PROJECT_DIR="/scratch/gpfs/mg7411/samedifferent"
DATA_DIR="${PROJECT_DIR}/data/meta_h5/pb"
OUTPUT_BASE_DIR="${PROJECT_DIR}/results/fomaml_experiments/arch_${ARCH}_seed_${SEED}"

# Ensure output and log directories exist
mkdir -p $OUTPUT_BASE_DIR
mkdir -p logs

# Run the python script
python "${PROJECT_DIR}/all_tasks/experiment_all_tasks_fomaml.py" \
    --architecture "$ARCH" \
    --seed "$SEED" \
    --data_dir "$DATA_DIR" \
    --output_base_dir "$OUTPUT_BASE_DIR" \
    --first_order \
    --meta_batch_size 32 \
    --adaptation_steps_train 5 \
    --adaptation_steps_val 5 \
    --adaptation_steps_test 15 \
    --patience 10

echo "Job finished with exit code $?" 