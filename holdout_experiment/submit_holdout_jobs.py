import os
import subprocess
from datetime import datetime
from .models import pb_tasks

def generate_slurm_script(task, output_dir):
    """Generate a SLURM script for training with a specific held-out task."""
    task_dir = os.path.join(output_dir, f"holdout_{task}")
    data_dir = "/scratch/gpfs/mg7411/data"  # Path to data directory in della containing pb/pb and svrt_fixed
    
    return f"""#!/bin/bash
#SBATCH --job-name=holdout_{task}
#SBATCH --output={task_dir}/slurm_%j.out
#SBATCH --error={task_dir}/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=pli
#SBATCH --account=nam
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8

# Load required modules
module purge
module load anaconda3/2023.3
module load cuda/11.7.1

# Activate conda environment
conda activate tensorflow

# Create task directory
mkdir -p {task_dir}

# Run training script
python -m train_holdout_seeds \\
    --task {task} \\
    --data_dir {data_dir} \\
    --output_dir {task_dir} \\
    --num_seeds 20
"""

def main():
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"exp3_holdout_runs_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating output directory: {output_dir}")
    print(f"Will submit jobs for tasks: {pb_tasks}")
    
    # Submit jobs for each held-out task
    job_ids = []
    for task in pb_tasks:
        # Create task directory
        task_dir = os.path.join(output_dir, f"holdout_{task}")
        os.makedirs(task_dir, exist_ok=True)
        
        # Generate and save SLURM script
        slurm_script = generate_slurm_script(task, output_dir)
        script_path = os.path.join(task_dir, "train.slurm")
        with open(script_path, "w") as f:
            f.write(slurm_script)
        
        print(f"\nSubmitting job for held-out task: {task}")
        print(f"Slurm script saved to: {script_path}")
        
        # Submit job
        result = subprocess.run(["sbatch", script_path], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            # Extract job ID from sbatch output
            job_id = result.stdout.strip().split()[-1]
            job_ids.append((task, job_id))
            print(f"Job submitted successfully (Job ID: {job_id})")
        else:
            print(f"Error submitting job for task {task}")
            print(f"Error message: {result.stderr}")
    
    # Save job IDs
    job_ids_file = os.path.join(output_dir, "job_ids.txt")
    with open(job_ids_file, "w") as f:
        for task, job_id in job_ids:
            f.write(f"Task {task}: {job_id}\n")
    
    print(f"\nAll jobs submitted. Output directory: {output_dir}")
    print(f"Job IDs saved to: {job_ids_file}")
    print("\nMonitor jobs with: squeue -u $USER")

if __name__ == "__main__":
    main() 