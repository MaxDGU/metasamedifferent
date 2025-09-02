#!/usr/bin/env python
# naturalistic/test_meta_model.py

import torch
import torch.nn as nn
# import torch.optim as optim # No longer strictly needed if MAML re-init is simple
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import random
import sys
import gc
import learn2learn as l2l
import json # Added for JSON output
import matplotlib.pyplot as plt # Added for plotting

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from baselines.models.conv2 import SameDifferentCNN as Conv2CNN
    from baselines.models.conv4 import SameDifferentCNN as Conv4CNN
    from baselines.models.conv6 import SameDifferentCNN as Conv6CNN
    print("Successfully imported Conv{2,4,6}lrCNN models from baselines.models")
except ImportError as e:
    print(f"Error importing models: {e}")
    print("Please ensure conv2.py, conv4.py, conv6.py are in baselines/models/ and project root is in PYTHONPATH.")
    exit(1)

ARCHITECTURES_CONFIG = {
    'conv2lr': {'class': Conv2CNN, 'seeds': [42, 123, 789, 555, 999]},
    'conv4lr': {'class': Conv4CNN, 'seeds': [42, 123, 789, 555, 999]},
    'conv6lr': {'class': Conv6CNN, 'seeds': [42, 123, 789, 555, 999]}
}

def accuracy(predictions, targets):
    """Binary accuracy: assumes predictions are logits [N, 2], targets are [N]."""
    predicted_labels = torch.argmax(predictions, dim=1)
    correct = (predicted_labels == targets).float()
    return correct.mean()

class MetaNaturalisticDataset(Dataset):
    """
    Dataset for loading meta-learning episodes from the naturalistic HDF5 files.
    (Copied from meta_naturalistic_train.py)
    """
    def __init__(self, h5_path, transform=None):
        self.h5_path = Path(h5_path)
        self.user_transform = transform 

        from torchvision import transforms as T 
        self.processing_transform = T.Compose([
            T.ToPILImage(), 
            T.Resize((128, 128)),
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        self._file = None 
        self.episode_keys = []
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")
        try:
            self._file = h5py.File(self.h5_path, 'r')
            self.episode_keys = sorted([k for k in self._file.keys() if k.startswith('episode_')])
            if not self.episode_keys:
                raise ValueError(f"No episode groups found in {self.h5_path}")
            # print(f"Loaded {len(self.episode_keys)} episodes from {self.h5_path.name}") # Less verbose for batch testing
        except Exception as e:
            print(f"Error opening or reading keys from {self.h5_path}: {e}")
            if self._file: self._file.close()
            raise

    def __len__(self):
        return len(self.episode_keys)

    def __getitem__(self, idx):
        if not hasattr(self, '_file') or not self._file:
             self._file = h5py.File(self.h5_path, 'r')
        episode_key = self.episode_keys[idx]
        try:
            ep_group = self._file[episode_key]
            support_images = ep_group['support_images'][()]
            support_labels = ep_group['support_labels'][()]
            query_images = ep_group['query_images'][()]
            query_labels = ep_group['query_labels'][()]
        except KeyError as e:
            print(f"Error accessing data for key {episode_key} in {self.h5_path}: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error reading episode {episode_key}: {e}")
            raise

        transformed_support_images = [self.processing_transform(img) for img in support_images]
        support_images_tensor = torch.stack(transformed_support_images)
        transformed_query_images = [self.processing_transform(img) for img in query_images]
        query_images_tensor = torch.stack(transformed_query_images)
        support_labels_tensor = torch.from_numpy(support_labels).long()
        query_labels_tensor = torch.from_numpy(query_labels).long() 

        if self.user_transform:
            support_images_tensor = torch.stack([self.user_transform(img) for img in support_images_tensor])
            query_images_tensor = torch.stack([self.user_transform(img) for img in query_images_tensor])
        return support_images_tensor, support_labels_tensor, query_images_tensor, query_labels_tensor

    def close(self):
        if hasattr(self, '_file') and self._file:
            self._file.close()
            self._file = None
    def __del__(self):
        self.close()

def fast_adapt(batch, learner, loss_fn, adaptation_steps, device, episode_idx_debug="N/A", force_first_order_adapt=False):
    support_images, support_labels, query_images, query_labels = batch
    support_images, support_labels = support_images.to(device), support_labels.to(device)
    query_images, query_labels = query_images.to(device), query_labels.to(device)

    original_learner_training_state = learner.training
    learner.train() 

    is_adapt_first_order = learner.first_order 
    if force_first_order_adapt:
        is_adapt_first_order = True
    
    for step in range(adaptation_steps):
        with torch.enable_grad(): 
            for p in learner.module.parameters(): 
                 p.requires_grad_(True)

            adaptation_logits = learner(support_images)
            adaptation_error = loss_fn(adaptation_logits, support_labels)
            if torch.isnan(adaptation_error) or torch.isinf(adaptation_error):
                print(f"WARNING (fast_adapt {episode_idx_debug}, Inner Step {step+1}): Unstable adaptation_error: {adaptation_error.item()}")
        
        learner.adapt(adaptation_error, first_order=is_adapt_first_order)

    learner.train(original_learner_training_state) 

    evaluation_logits = learner(query_images)
    evaluation_error = loss_fn(evaluation_logits, query_labels)
    evaluation_acc = accuracy(evaluation_logits, query_labels)

    if torch.isnan(evaluation_error) or torch.isinf(evaluation_error):
        print(f"WARNING (fast_adapt {episode_idx_debug}): Unstable evaluation_error: {evaluation_error.item()}")

    return evaluation_error, evaluation_acc

def parse_args():
    parser = argparse.ArgumentParser(description="Test trained MAML models, aggregate results, and plot.")
    # Paths
    parser.add_argument('--data_dir', type=str, default='/scratch/gpfs/mg7411/samedifferent/data/naturalistic', help='Directory containing test.h5')
    parser.add_argument('--base_log_dir', type=str, default='/scratch/gpfs/mg7411/samedifferent/logs_naturalistic_meta', help='Base directory where logs for runs are stored')
    parser.add_argument('--output_json_path', type=str, default='test_results/meta_summary.json', help='Path to save aggregated results in JSON format.')
    parser.add_argument('--output_plot_path', type=str, default='test_results/meta_summary_plot.png', help='Path to save the summary plot.')

    # Meta-Learning Hyperparameters (must match training for loading)
    parser.add_argument('--inner_lr', type=float, default=1e-5, help='Inner loop learning rate (must match training).')
    parser.add_argument('--inner_steps', type=int, default=5, help='Number of adaptation steps (must match training).')
    # parser.add_argument('--first_order', action='store_true', help='This flag is now determined from checkpoint if possible, otherwise assumes False for wrapper init.')


    # Testing Settings
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu)')
    parser.add_argument('--max_test_episodes', type=int, default=None, help='Maximum number of episodes for testing. Default: all.')
    parser.add_argument('--amp_dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16'], help='AMP dtype if used.')
    return parser.parse_args()

def run_single_test(arch_name, seed, model_class, args):
    """Loads a single model and runs the test set, returning accuracy."""
    device = torch.device(args.device)
    
    # Corrected path construction based on user feedback and SLURM script
    # LOG_BASE_DIR / ARCH / seed_SEED / ARCH / seed_SEED / ARCH_best.pth
    # args.log_dir in train script was: LOG_BASE_DIR / ARCH / seed_SEED
    # then in train script log_dir was: Path(args.log_dir) / args.model / f"seed_{args.seed}"
    # So, actual_log_dir_for_checkpoint = Path(args.base_log_dir) / arch_name / f"seed_{seed}" / arch_name / f"seed_{seed}"
    
    actual_log_dir_for_checkpoint = Path(args.base_log_dir) / arch_name / f"seed_{seed}" / arch_name / f"seed_{seed}"
    best_model_path = actual_log_dir_for_checkpoint / f"{arch_name}_best.pth"

    print(f"\nAttempting to test: Arch={arch_name}, Seed={seed}")
    print(f"Looking for model at: {best_model_path}")

    if not best_model_path.exists():
        print(f"ERROR: Best model checkpoint not found at {best_model_path}. Skipping.")
        return None

    model = model_class().to(device)
    
    # Determine first_order for MAML wrapper init from checkpoint if possible
    # Default to False if not found in checkpoint args, can be overridden by a specific script arg if added back
    checkpoint_first_order = False # Default
    try:
        # Peek into checkpoint for args without loading everything yet, if necessary, or just load
        temp_checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False) # Load to CPU first
        if 'args' in temp_checkpoint and hasattr(temp_checkpoint['args'], 'first_order'):
            checkpoint_first_order = temp_checkpoint['args'].first_order
            print(f"  Checkpoint trained with first_order: {checkpoint_first_order}")
        else:
            print("  Could not determine first_order from checkpoint args, assuming False for MAML wrapper init.")
        del temp_checkpoint # Free memory
    except Exception as e:
        print(f"  Warning: Could not peek into checkpoint for first_order arg: {e}. Assuming False.")


    maml = l2l.algorithms.MAML(
        model,
        lr=args.inner_lr,
        first_order=checkpoint_first_order, # Use determined value
        allow_unused=True,
        allow_nograd=True 
    ).to(device)

    try:
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False) 
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'maml_state_dict' in checkpoint:
            maml.load_state_dict(checkpoint['maml_state_dict'])
        else:
            print("  MAML state_dict not found in checkpoint. Wrapper will use loaded base model weights.")
        
        if 'args' in checkpoint:
            train_args = checkpoint['args']
            print(f"  Successfully loaded. Model trained with args: {train_args}")
            if train_args.inner_lr != args.inner_lr: print(f"  WARNING: Test inner_lr ({args.inner_lr}) vs Train ({train_args.inner_lr})")
            if train_args.inner_steps != args.inner_steps: print(f"  WARNING: Test inner_steps ({args.inner_steps}) vs Train ({train_args.inner_steps})")
            # Compare actual first_order used for wrapper vs one in file
            if hasattr(train_args, 'first_order') and train_args.first_order != checkpoint_first_order:
                 print(f"  WARNING: MAML wrapper init first_order ({checkpoint_first_order}) vs Train first_order in args ({train_args.first_order}). This might be an issue if they differ.")
        else:
            print("  Successfully loaded. No 'args' in checkpoint to verify against.")


    except Exception as e:
        print(f"Error loading checkpoint for {arch_name} seed {seed}: {e}. Skipping.")
        return None

    # --- Load Test Data ---
    transform = None 
    test_h5_path = Path(args.data_dir) / 'test.h5'
    if not test_h5_path.exists():
        print(f"Test data file not found: {test_h5_path}. Skipping.")
        return None
    
    try:
        test_dataset = MetaNaturalisticDataset(test_h5_path, transform=transform)
        num_test_episodes_to_run = len(test_dataset)
        if args.max_test_episodes is not None and args.max_test_episodes > 0:
            num_test_episodes_to_run = min(len(test_dataset), args.max_test_episodes)
        
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
        print(f"  Testing on {num_test_episodes_to_run} episodes from {test_h5_path.name}")

    except Exception as e:
        print(f"Error loading test dataset: {e}. Skipping.")
        if 'test_dataset' in locals() and hasattr(test_dataset, 'close'): test_dataset.close()
        return None

    maml.eval()
    criterion = nn.CrossEntropyLoss()
    meta_test_loss = 0.0
    meta_test_acc = 0.0
    test_episodes_processed = 0
    
    is_cuda_and_amp_enabled = args.device == 'cuda' and torch.cuda.is_available()
    amp_dtype_val = torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16

    pbar_test = tqdm(test_loader, total=num_test_episodes_to_run, desc=f"Testing {arch_name} seed {seed}", leave=False)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar_test):
            if batch_idx >= num_test_episodes_to_run: break
            task_batch = [d.squeeze(0) for d in batch] 
            
            with torch.amp.autocast(device_type=args.device if is_cuda_and_amp_enabled else 'cpu', 
                                    dtype=amp_dtype_val if is_cuda_and_amp_enabled else torch.float32, 
                                    enabled=is_cuda_and_amp_enabled):
                learner = maml.clone()
                evaluation_loss, evaluation_acc = fast_adapt(task_batch, learner, criterion, 
                                                               args.inner_steps, device,
                                                               episode_idx_debug=f"test_ep_{batch_idx}",
                                                               force_first_order_adapt=True) 
            
            if torch.isnan(evaluation_loss) or torch.isinf(evaluation_loss):
                print(f"ERROR: NaN/Inf loss in test ep {batch_idx} for {arch_name} seed {seed}. Skipping.")
                del learner, task_batch
                if device == 'cuda': torch.cuda.empty_cache()
                continue

            meta_test_loss += evaluation_loss.item()
            meta_test_acc += evaluation_acc.item()
            test_episodes_processed += 1
            pbar_test.set_postfix(loss=f"{evaluation_loss.item():.4f}", acc=f"{evaluation_acc.item():.3f}")

            del learner, task_batch, evaluation_loss, evaluation_acc
            if device == 'cuda': torch.cuda.empty_cache()

    avg_test_acc_val = 0
    if test_episodes_processed > 0:
        # avg_test_loss = meta_test_loss / test_episodes_processed
        avg_test_acc_val = meta_test_acc / test_episodes_processed
        print(f"  Test Result for {arch_name} seed {seed}: Avg Acc: {avg_test_acc_val:.4f}")
    else:
        print(f"  No episodes processed for {arch_name} seed {seed}.")
    
    gc.collect()
    if hasattr(test_dataset, 'close'): test_dataset.close()
    return avg_test_acc_val if test_episodes_processed > 0 else None


def main():
    args = parse_args()
    
    # Create output directories if they don't exist
    Path(args.output_json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_plot_path).parent.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for arch_name, config in ARCHITECTURES_CONFIG.items():
        model_class = config['class']
        seeds = config['seeds']
        arch_accuracies = []
        
        print(f"\n--- Processing Architecture: {arch_name} ---")
        for seed in seeds:
            # Set seed for MAML operations like clone, though main determinism comes from loaded weights
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            test_acc = run_single_test(arch_name, seed, model_class, args)
            if test_acc is not None:
                arch_accuracies.append(test_acc)
        
        if arch_accuracies:
            mean_acc = np.mean(arch_accuracies)
            std_acc = np.std(arch_accuracies)
            all_results[arch_name] = {
                "accuracies": arch_accuracies,
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "num_seeds_successful": len(arch_accuracies)
            }
            print(f"Summary for {arch_name}: Mean Acc: {mean_acc:.4f} +/- {std_acc:.4f} (from {len(arch_accuracies)} seeds)")
        else:
            all_results[arch_name] = {
                "accuracies": [],
                "mean_accuracy": 0.0,
                "std_accuracy": 0.0,
                "num_seeds_successful": 0
            }
            print(f"No successful test runs for architecture {arch_name}")

    # Save results to JSON
    print(f"\nSaving aggregated results to {args.output_json_path}")
    with open(args.output_json_path, 'w') as f:
        json.dump(all_results, f, indent=4)

    # Plot results
    arch_names_plot = list(all_results.keys())
    mean_accuracies_plot = [all_results[arch].get('mean_accuracy', 0) for arch in arch_names_plot]
    std_accuracies_plot = [all_results[arch].get('std_accuracy', 0) for arch in arch_names_plot]

    if not any(mean_accuracies_plot): # Check if all means are zero (e.g. no successful runs)
        print("No data to plot or all accuracies are zero. Skipping plot generation.")
    else:
        plt.figure(figsize=(10, 6))
        plt.bar(arch_names_plot, mean_accuracies_plot, yerr=std_accuracies_plot, capsize=5, color=['skyblue', 'lightcoral', 'lightgreen'])
        plt.xlabel("Model Architecture")
        plt.ylabel("Mean Test Accuracy")
        plt.title("Mean Test Accuracy by Architecture (Meta-Learning)")
        plt.ylim(0, 1) # Assuming accuracy is between 0 and 1
        
        # Add text labels for mean accuracies on top of bars
        for i, val in enumerate(mean_accuracies_plot):
            plt.text(i, val + std_accuracies_plot[i] + 0.02, f"{val:.3f}", ha='center', va='bottom')

        plt.savefig(args.output_plot_path)
        print(f"Saved plot to {args.output_plot_path}")
        # plt.show() # Optionally show plot if run in an interactive environment

    print("\nTesting and aggregation finished.")

if __name__ == "__main__":
    main()

 