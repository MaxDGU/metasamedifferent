#!/bin/bash
#SBATCH --job-name=pb_leave_one_out
#SBATCH --output=logs/pb_leave_one_out_%A_%a.out
#SBATCH --error=logs/pb_leave_one_out_%A_%a.err
#SBATCH --array=0-9
#SBATCH --ntasks1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=nam
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mg7411@princeton.edu



# Load necessary modules (modify according to your cluster setup)
module purge
module load anaconda3
module load cuda/11.8
module load cudnn/8.6.0

# Activate conda environment
source activate tensorflow

# Create logs directory
mkdir -p logs

# Define the tasks array
TASKS=(
    "regular"
    "lines"
    "open"
    "wider_line"
    "scrambled"
    "random_color"
    "arrows"
    "irregular"
    "filled"
    "original"
)

# Get the current task based on array job ID
HELD_OUT_TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}

# Set environment variables for better performance
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0

# Create output directory for this task
OUTPUT_DIR="results/leave_one_out/${HELD_OUT_TASK}"
mkdir -p $OUTPUT_DIR

# Print job information
echo "Starting job for held out task: ${HELD_OUT_TASK}"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Output directory: $OUTPUT_DIR"
echo "GPU device: $CUDA_VISIBLE_DEVICES"
echo "Training adaptation steps: 5"
echo "Testing adaptation steps: 15"

# Run the experiment with updated learning rates and test adaptation steps
python experiment1_leave_one_out.py \
    --pb_data_dir data/pb/pb \
    --output_dir $OUTPUT_DIR \
    --held_out_task $HELD_OUT_TASK \
    --batch_size 16 \
    --epochs 100 \
    --inner_lr 0.01 \
    --outer_lr 0.0001 \
    --adaptation_steps 5 \
    --test_adaptation_steps 15 \
    --seed 42

# Print completion message
echo "Job completed for held out task: ${HELD_OUT_TASK}" 