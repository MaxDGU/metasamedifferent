import os
import subprocess
import json
import numpy as np
from itertools import product

# Define tasks and architectures
from models.utils import PB_TASKS, ARCHITECTURES, SEEDS

def create_array_script(output_dir):
    """Create a Slurm array script for all jobs."""
    script = f"""#!/bin/bash
#SBATCH --job-name=pb_array
#SBATCH --output=results/pb_baselines/slurm_%A_%a.out
#SBATCH --error=results/pb_baselines/slurm_%A_%a.err
#SBATCH --time=24:00:00  # Increased time since training on all tasks
#SBATCH --mem=32G       # Increased memory since training on all tasks
#SBATCH --partition=pli
#SBATCH --account=nam
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-149  # 10 test tasks * 3 architectures * 5 seeds = 150 jobs

# Load required modules
module load anaconda3
module load cuda/11.7.0

# Activate conda environment
source activate torch_env

# Calculate task, architecture, and seed from array index
declare -a TASKS=("regular" "lines" "open" "wider_line" "scrambled" "random_color" "arrows" "irregular" "filled" "original")
declare -a ARCHS=("conv2" "conv4" "conv6")
declare -a SEEDS=(47 48 49 50 51)  # Updated seeds

task_idx=$((SLURM_ARRAY_TASK_ID / 15))
arch_idx=$(((SLURM_ARRAY_TASK_ID % 15) / 5))
seed_idx=$((SLURM_ARRAY_TASK_ID % 5))

TASK=${{TASKS[$task_idx]}}
ARCH=${{ARCHS[$arch_idx]}}
SEED=${{SEEDS[$seed_idx]}}

echo "Running job for test_task=$TASK arch=$ARCH seed=$SEED"

# Run the training script
python -m train_pb_models \\
    --test_task $TASK \\
    --architecture $ARCH \\
    --seed $SEED \\
    --data_dir data/pb/pb \\
    --output_dir {output_dir} \\
    --epochs 100 \\
    --patience 3 \\
    --val_freq 10 \\
    --improvement_threshold 0.02
"""
    
    script_path = os.path.join(output_dir, 'array_job.sh')
    os.makedirs(output_dir, exist_ok=True)
    with open(script_path, 'w') as f:
        f.write(script)
    
    return script_path

def submit_jobs(output_dir='results/pb_baselines', tasks=None, architectures=None, seeds=None):
    """Submit jobs as a Slurm array."""
    if tasks is None:
        tasks = PB_TASKS
    if architectures is None:
        architectures = ARCHITECTURES
    if seeds is None:
        seeds = SEEDS
    
    # Calculate array indices for specified combinations
    all_combinations = list(product(tasks, architectures, seeds))
    indices = []
    for task, arch, seed in all_combinations:
        task_idx = PB_TASKS.index(task)
        arch_idx = ARCHITECTURES.index(arch)
        seed_idx = SEEDS.index(seed)
        array_idx = task_idx * 15 + arch_idx * 5 + seed_idx
        indices.append(array_idx)
    
    # Create array script
    script_path = create_array_script(output_dir)
    
    # Submit array job with specific indices
    array_spec = ','.join(map(str, sorted(indices)))
    try:
        cmd = ['sbatch', f'--array={array_spec}', script_path]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        array_job_id = result.stdout.strip().split()[-1]
        print(f'Submitted array job {array_job_id} with {len(indices)} tasks')
        return array_job_id
    except subprocess.CalledProcessError as e:
        print(f'Error submitting array job: {e}')
        return None

def check_job_status(job_id):
    """Check if a Slurm array job has completed."""
    try:
        result = subprocess.run(['squeue', '-j', job_id], capture_output=True, text=True)
        # If job not found in queue, it's completed
        return 'slurm_load_jobs error' in result.stderr
    except subprocess.CalledProcessError:
        return True  # Assume completed if squeue fails

def compile_results(output_dir='results/pb_baselines', save_path=None):
    """Compile results from completed jobs."""
    results = {}
    missing_results = False
    
    for task in PB_TASKS:
        results[task] = {}
        for arch in ARCHITECTURES:
            accuracies = []
            for seed in SEEDS:
                # Updated path to reflect new directory structure
                metrics_path = os.path.join(output_dir, 'all_tasks', arch, f'test_{task}', f'seed_{seed}', 'metrics.json')
                try:
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                        accuracies.append(metrics['test_acc'])
                except FileNotFoundError:
                    print(f'Warning: No results found for {task} {arch} seed {seed}')
                    missing_results = True
            
            if accuracies:
                results[task][arch] = {
                    'mean': float(np.mean(accuracies)),
                    'std': float(np.std(accuracies)),
                    'individual_accuracies': accuracies
                }
    
    if save_path and not missing_results:
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results, not missing_results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='results/pb_baselines')
    parser.add_argument('--compile_only', action='store_true', 
                      help='Only compile results without submitting new jobs')
    parser.add_argument('--tasks', nargs='+', help='Specific tasks to run (default: all)')
    parser.add_argument('--architectures', nargs='+', help='Specific architectures to run (default: all)')
    parser.add_argument('--seeds', type=int, nargs='+', help='Specific seeds to use (default: all)')
    parser.add_argument('--wait', action='store_true',
                      help='Wait for jobs to complete before compiling results')
    args = parser.parse_args()
    
    if not args.compile_only:
        # Submit array job
        array_job_id = submit_jobs(
            output_dir=args.output_dir,
            tasks=args.tasks,
            architectures=args.architectures,
            seeds=args.seeds
        )
        
        if args.wait and array_job_id:
            print("Waiting for jobs to complete...")
            while not check_job_status(array_job_id):
                import time
                time.sleep(60)  # Check every minute
    
    # Compile and save results
    results, complete = compile_results(
        output_dir=args.output_dir,
        save_path=os.path.join(args.output_dir, 'compiled_results.json')
    )
    
    if complete:
        # Print summary
        print('\nResults Summary:')
        for task in sorted(results.keys()):
            print(f'\n{task}:')
            for arch in sorted(results[task].keys()):
                mean = results[task][arch]['mean'] * 100
                std = results[task][arch]['std'] * 100
                print(f'  {arch}: {mean:.2f}% ± {std:.2f}%')
    else:
        print("\nSome results are missing. Please wait for all jobs to complete before compiling results.")
        print("You can run with --compile_only later to get the final results.") 