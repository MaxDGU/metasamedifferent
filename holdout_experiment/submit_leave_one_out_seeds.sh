#!/bin/bash
#SBATCH --job-name=pb_loo_seeds
#SBATCH --output=slurm_logs/loo_seeds_%A_%a.out
#SBATCH --error=slurm_logs/loo_seeds_%A_%a.err
#SBATCH --array=0-99  # 10 seeds * 10 tasks = 100 jobs
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mg7411@princeton.edu

# Create logs directory
mkdir -p slurm_logs

# Array of PB tasks and seeds
PB_TASKS=(regular lines open wider_line scrambled random_color arrows irregular filled original)
SEEDS=(42 43 44 45 46 47 48 49 50 51)

# Calculate task and seed indices
TASK_IDX=$((SLURM_ARRAY_TASK_ID % 10))
SEED_IDX=$((SLURM_ARRAY_TASK_ID / 10))

# Get task and seed for this job
TASK=${PB_TASKS[$TASK_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "Running with task: $TASK, seed: $SEED"

# Run the experiment
python experiment1_leave_one_out.py \
    --pb_data_dir data/pb/pb \
    --output_dir results/leave_one_out_seeds/$TASK \
    --held_out_task $TASK \
    --seed $SEED \
    --batch_size 16 \
    --epochs 1000 \
    --inner_lr 0.01 \
    --outer_lr 0.0001 \
    --adaptation_steps 5 \
    --test_adaptation_steps 15 