#!/bin/bash
#SBATCH --job-name=conv6_baselines
#SBATCH --output=conv6_svrt_%A_%a.out
#SBATCH --error=conv6_svrt_%A_%a.err
#SBATCH --array=0-9
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=nam
#SBATCH --mem=20G
#SBATCH --time=8:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mg7411@princeton.edu

# Load modules - update these to match your system's available modules
module purge  # Clear any previously loaded modules
module load anaconda3  # Or whatever Python module your system uses
module load cuda/11.7  # Update version as needed

# Activate conda environment if needed
conda activate tensorflow

# Calculate seed
SEED=$((42 + SLURM_ARRAY_TASK_ID))

# Run the training
python conv6.py --seed $SEED

# After all jobs complete, run analysis script to average results
if [ $SLURM_ARRAY_TASK_ID -eq 9 ]; then
    echo "Starting analysis script..."
    sleep 60  # Wait a minute to ensure all files are saved
    
    python - <<EOF
import glob
import json
import numpy as np
from collections import defaultdict
import os
import time

def combine_results():
    # Wait for files to appear
    max_wait = 300  # Maximum wait time in seconds
    wait_time = 0
    while wait_time < max_wait:
        json_files = glob.glob('conv6_all_problems_results_seed*.json')
        if len(json_files) >= 10:  # We expect 10 files
            break
        print(f"Found {len(json_files)} files, waiting for more...")
        time.sleep(30)
        wait_time += 30
    
    # Dictionary to store all results
    all_results = defaultdict(lambda: defaultdict(list))
    
    # First try to read JSON files
    json_files = sorted(glob.glob('conv6_all_problems_results_seed*.json'))
    print(f"\nFound {len(json_files)} JSON files:")
    for f in json_files:
        print(f"  {f}")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                results = json.load(f)
                seed = int(json_file.split('seed')[-1].split('.')[0])
                print(f"\nProcessing results from seed {seed}")
                print(f"Keys in results: {list(results.keys())}")
                for problem, metrics in results.items():
                    print(f"  Problem {problem} metrics: {list(metrics.keys())}")
                    for metric_name, value in metrics.items():
                        all_results[problem][metric_name].append(value)
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
            print(f"File contents: {open(json_file).read()}")
    
    if not all_results:
        print("No results found! Check if JSON files exist and are readable.")
        return
    
    print("\nCollected results:")
    for problem in all_results:
        print(f"\nProblem {problem}:")
        for metric, values in all_results[problem].items():
            print(f"  {metric}: {len(values)} values")
    
    # Calculate statistics
    final_results = {}
    for problem in all_results:
        final_results[problem] = {
            metric: {
                'mean': float(np.mean(values)),  # Convert to float for JSON serialization
                'std': float(np.std(values)),
                'max': float(np.max(values)),
                'min': float(np.min(values)),
                'values': values  # Keep individual values for reference
            }
            for metric, values in all_results[problem].items()
        }
    
    # Save detailed results
    print("\nSaving final results...")
    with open('conv6_final_results_detailed.json', 'w') as f:
        json.dump(final_results, f, indent=4)
    print("Results saved successfully!")
    
    # Print summary
    print("\nFinal Results Summary:")
    print("="*50)
    for problem in sorted(final_results.keys()):
        print(f"\nProblem {problem}:")
        metrics = final_results[problem]
        print(f"Validation Accuracy: {metrics['val_acc']['mean']:.4f} ± {metrics['val_acc']['std']:.4f}")
        print(f"  Max: {metrics['val_acc']['max']:.4f}")
        print(f"  Min: {metrics['val_acc']['min']:.4f}")
        print(f"Validation Loss: {metrics['val_loss']['mean']:.4f} ± {metrics['val_loss']['std']:.4f}")
        print(f"Best Validation Accuracy: {metrics['best_val_acc']['mean']:.4f} ± {metrics['best_val_acc']['std']:.4f}")
        print(f"Best Train Accuracy: {metrics['best_train_acc']['mean']:.4f} ± {metrics['best_train_acc']['std']:.4f}")
        print(f"Average Epochs: {metrics['epochs_trained']['mean']:.1f} ± {metrics['epochs_trained']['std']:.1f}")

print("Starting results combination...")
combine_results()
print("Analysis complete!")
EOF
fi 