import os
import torch
import json
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import learn2learn as l2l
import torch.nn.functional as F
import copy
import gc
from models.conv6lr import SameDifferentCNN, SameDifferentDataset
from torch.utils.data import DataLoader 
import sys

# Define PB tasks
PB_TASKS = [
    'regular', 'lines', 'open', 'wider_line', 'scrambled',
    'random_color', 'arrows', 'irregular', 'filled', 'original'
]

def accuracy(predictions, targets):
    """Calculate binary classification accuracy."""
    predicted_labels = (predictions[:, 1] > 0.0).float()
    return (predicted_labels == targets).float().mean()

def test_model(model, test_loader, device, test_adaptation_steps, inner_lr=None):
    """Test function with better error handling and debugging"""
    try:
        model.eval()
        total_loss = 0
        total_acc = 0
        num_batches = 0
        
        print(f"\nTesting with {test_adaptation_steps} adaptation steps")
        print(f"Dataset size: {len(test_loader.dataset)} episodes")
        
        # Use tqdm for progress tracking
        for batch in tqdm(test_loader, desc="Testing episodes"):
            try:
                # Clone model for this episode
                adapted_model = copy.deepcopy(model)
                adapted_model.train()
                
                # Move data to GPU and verify shapes
                support_images = batch['support_images'].to(device, non_blocking=True)
                support_labels = batch['support_labels'].to(device, non_blocking=True)
                query_images = batch['query_images'].to(device, non_blocking=True)
                query_labels = batch['query_labels'].to(device, non_blocking=True)
                
                # Print shapes for debugging
                print(f"Shapes - Support: {support_images.shape}, Labels: {support_labels.shape}")
                print(f"       Query: {query_images.shape}, Labels: {query_labels.shape}")
                
                # Get dimensions
                B, S, C, H, W = support_images.shape
                _, Q, _, _, _ = query_images.shape
                
                # Reshape support images and labels
                support_images = support_images.view(B*S, C, H, W)  # [B*S, C, H, W]
                support_labels = support_labels.view(B*S)           # [B*S]
                
                # Reshape query images and labels
                query_images = query_images.view(B*Q, C, H, W)      # [B*Q, C, H, W]
                query_labels = query_labels.view(B*Q)               # [B*Q]
                
                # Get per-layer learning rates
                layer_lrs = adapted_model.get_layer_lrs()
                
                # Support set adaptation
                for step in range(test_adaptation_steps):
                    support_preds = adapted_model(support_images)
                    support_loss = F.binary_cross_entropy_with_logits(
                        support_preds[:, 1], support_labels.float())
                    
                    if torch.isnan(support_loss):
                        print(f"WARNING: NaN support loss at step {step}")
                        continue
                    
                    grads = torch.autograd.grad(
                        support_loss,
                        adapted_model.parameters(),
                        create_graph=True,
                        allow_unused=True
                    )
                    
                    # Apply per-layer learning rates
                    for (name, param), grad in zip(adapted_model.named_parameters(), grads):
                        if grad is not None:
                            lr = layer_lrs.get(name, torch.tensor(0.01).to(device))
                            param.data = param.data - lr.abs() * grad
                
                # Evaluate on query set
                adapted_model.eval()
                with torch.no_grad():
                    query_preds = adapted_model(query_images)
                    query_loss = F.binary_cross_entropy_with_logits(
                        query_preds[:, 1], query_labels.float())
                    
                    if torch.isnan(query_loss):
                        print("WARNING: NaN query loss")
                        continue
                        
                    query_acc = accuracy(query_preds, query_labels)
                    
                    if torch.isnan(query_acc):
                        print("WARNING: NaN accuracy")
                        continue
                    
                    total_loss += query_loss.item()
                    total_acc += query_acc.item()
                    num_batches += 1
                    
                    print(f"Batch {num_batches} - Loss: {query_loss.item():.4f}, Acc: {query_acc.item():.4f}")
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"WARNING: GPU OOM error. Trying to recover...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                    continue
                else:
                    print(f"ERROR: Runtime error in batch: {str(e)}")
                    sys.exit(1)
            except Exception as e:
                print(f"ERROR: Unexpected error in batch: {str(e)}")
                sys.exit(1)
            
            # Clear GPU memory after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if num_batches == 0:
            print("ERROR: No valid batches processed!")
            return None
        
        # Calculate final metrics
        final_loss = total_loss / num_batches
        final_acc = total_acc / num_batches
        
        print(f"\nTest Results:")
        print(f"Processed {num_batches} valid batches")
        print(f"Final Loss: {final_loss:.4f}")
        print(f"Final Accuracy: {final_acc:.4f}")
        
        return {
            'loss': float(final_loss),
            'accuracy': float(final_acc),
            'num_batches': num_batches
        }
    
    except Exception as e:
        print(f"ERROR: Test failed with error: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, 
                        default='exp3_holdout_runs_20250131_155132',
                        help='Directory containing the experiment checkpoints')
    parser.add_argument('--data_dir', type=str,
                        default='data/pb/pb',
                        help='Directory containing the PB test data')
    parser.add_argument('--output_dir', type=str,
                        default='results/experiment3',
                        help='Directory to save test results')
    parser.add_argument('--test_adaptation_steps', type=int, default=15,
                        help='Number of adaptation steps during testing')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for testing')
    args = parser.parse_args()
    
    # Setup device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPU access.")
    device = torch.device('cuda')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Store all results
    all_results = {}
    
    # Support set sizes to test
    support_sizes = [4, 6, 8, 10]
    
    try:
        # Test each task
        for task in tqdm(PB_TASKS, desc='Testing tasks'):
            print(f"\n{'='*50}")
            print(f"Testing task: {task}")
            print(f"{'='*50}")
            
            all_results[task] = {}
            
            # Construct paths
            model_dir = os.path.join(args.exp_dir, f'holdout_{task}', 'seed0')
            model_path = os.path.join(model_dir, f'model_{task}_final.pt')
            
            if not os.path.exists(model_path):
                print(f'Warning: No checkpoint found for task {task} at {model_path}')
                continue
            
            # Load model
            model = SameDifferentCNN().to(device)
            
            # First load the base model state dict
            # Remove learning rate parameters from state dict
            state_dict = {k: v for k, v in torch.load(model_path, map_location=device)['model_state_dict'].items() 
                         if not k.startswith('lr_')}
            
            # Load state dict with strict=False to ignore missing temperature parameter
            model.load_state_dict(state_dict, strict=False)
            
            # Manually set temperature parameter if needed
            if hasattr(model, 'temperature'):
                model.temperature = torch.nn.Parameter(torch.ones(1).to(device))
            
            model.eval()
            
            # Test each support set size separately
            for support_size in support_sizes:
                print(f"\nTesting with support size: {support_size}")
                
                # Create test dataset for this support size
                test_dataset = SameDifferentDataset(
                    args.data_dir,
                    [task],
                    'test',
                    support_sizes=[support_size]  # Only one support size at a time
                )
                
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True
                )
                
                # Test model
                results = test_model(
                    model,
                    test_loader,
                    device,
                    args.test_adaptation_steps,
                    None  # No need for inner_lr since we're using per-layer learning rates
                )
                
                if results is not None:
                    all_results[task][f'support_{support_size}'] = results
                    print(f"Support size {support_size} - Accuracy: {results['accuracy']*100:.2f}%")
            
            # Save results for this task
            task_results_path = os.path.join(args.output_dir, f'{task}_results.json')
            with open(task_results_path, 'w') as f:
                json.dump(all_results[task], f, indent=2)
        
        # Calculate summary statistics
        summary_stats = {}
        for task in PB_TASKS:
            if task in all_results:
                accuracies = []
                for support_size in support_sizes:
                    if f'support_{support_size}' in all_results[task]:
                        accuracies.append(all_results[task][f'support_{support_size}']['accuracy'])
                
                if accuracies:
                    summary_stats[task] = {
                        'mean_accuracy': float(np.mean(accuracies)),
                        'std_accuracy': float(np.std(accuracies)),
                        'num_support_sizes': len(accuracies)
                    }
        
        # Save all results
        with open(os.path.join(args.output_dir, 'all_results.json'), 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Save summary stats
        with open(os.path.join(args.output_dir, 'summary_stats.json'), 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print('\nTesting complete! Results saved to:', args.output_dir)
    
    except Exception as e:
        print(f"\nERROR: Testing failed with error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 