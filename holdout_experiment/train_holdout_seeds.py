import sys
import os
import argparse
import json
from models import main, pb_tasks

def train_holdout(held_out_task, data_dir, output_dir, num_seeds=20):
    """Train models with the specified held-out task for multiple seeds."""
    # Create output directory for this held-out task
    os.makedirs(output_dir, exist_ok=True)
    
    # Train models for each seed
    for seed in range(num_seeds):
        # Set up model save path for this seed
        seed_dir = os.path.join(output_dir, f'seed{seed}')
        os.makedirs(seed_dir, exist_ok=True)
        save_path = os.path.join(seed_dir, f'model_{held_out_task}_final.pt')
        
        # Train model with this held-out task and seed
        print(f"\nTraining with held-out task: {held_out_task}, seed: {seed}")
        print(f"Save path: {save_path}")
        print(f"Data directory: {data_dir}")
        
        metrics = main(
            held_out_task=held_out_task,
            seed=seed,
            save_path=save_path,
            data_dir=data_dir,
            return_metrics=True
        )
        
        # Save metrics for this seed
        metrics_path = os.path.join(seed_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='Task to hold out')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--num_seeds', type=int, default=10, help='Number of seeds to run')
    args = parser.parse_args()
    
    # Validate task
    if args.task not in pb_tasks:
        print(f"Error: {args.task} is not a valid PB task")
        print(f"Available tasks: {pb_tasks}")
        sys.exit(1)
    
    # Train the model
    train_holdout(
        held_out_task=args.task,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_seeds=args.num_seeds
    ) 