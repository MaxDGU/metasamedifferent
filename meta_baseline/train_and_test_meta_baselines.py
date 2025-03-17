import os
import torch
import torch.nn.functional as F
import json
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import copy
import gc
from pathlib import Path
import sys
import learn2learn as l2l
import torch.nn as nn

from conv2lr import SameDifferentCNN as Conv2CNN
from conv4lr import SameDifferentCNN as Conv4CNN
from conv6lr import SameDifferentCNN as Conv6CNN
from conv6lr import SameDifferentDataset, collate_episodes
from utils_meta import train_epoch, validate

PB_TASKS = [
    'regular', 'lines', 'open', 'wider_line', 'scrambled',
    'random_color', 'arrows', 'irregular', 'filled', 'original'
]
ARCHITECTURES = {
    'conv2': Conv2CNN,
    'conv4': Conv4CNN,
    'conv6': Conv6CNN
}

def accuracy(predictions, targets):
    """binary classification accuracy - predictions are list of logits, targets are list of labels"""
    predicted_labels = (predictions[:, 1] > 0.0).float()
    return (predicted_labels == targets.squeeze(1)).float().mean()

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/pb/pb',
                        help='Directory containing the PB dataset')
    parser.add_argument('--output_dir', type=str, default='results/meta_baselines',
                        help='Directory to save results')
    parser.add_argument('--architecture', type=str, required=True,
                        choices=['conv2', 'conv4', 'conv6'],
                        help='Model architecture to use')
    parser.add_argument('--seed', type=int, required=True,
                        help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training and testing')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--support_size', type=int, default=10,
                        help='Number of support examples per class')
    parser.add_argument('--adaptation_steps', type=int, default=5,
                        help='Number of adaptation steps during training')
    parser.add_argument('--test_adaptation_steps', type=int, default=15,
                        help='Number of adaptation steps during testing')
    parser.add_argument('--inner_lr', type=float, default=0.05,
                        help='Inner loop learning rate')
    parser.add_argument('--outer_lr', type=float, default=0.001,
                        help='Outer loop learning rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--min_delta', type=float, default=0.001,
                        help='Minimum improvement for early stopping')
    args = parser.parse_args()
    
    try:
        # Check for CUDA
        if not torch.cuda.is_available():
            print("WARNING: CUDA is not available. Running on CPU, but this will be slow and may not work correctly.")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
            # Set CUDA memory management settings
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = False  # More memory efficient
            torch.backends.cudnn.deterministic = True
            # Set CUDA memory allocation to be more conservative
            torch.cuda.set_per_process_memory_fraction(0.8)  # Use only 80% of available GPU memory
        
        if not os.path.exists(args.data_dir):
            raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
        
        set_seed(args.seed)
        
        arch_dir = os.path.join(args.output_dir, args.architecture, f'seed_{args.seed}')
        os.makedirs(arch_dir, exist_ok=True)
        
        print("\nCreating datasets...")
        train_dataset = SameDifferentDataset(args.data_dir, PB_TASKS, 'train', support_sizes=[args.support_size])
        val_dataset = SameDifferentDataset(args.data_dir, PB_TASKS, 'val', support_sizes=[args.support_size])
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                num_workers=1, pin_memory=True,
                                collate_fn=collate_episodes)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                              num_workers=1, pin_memory=True,
                              collate_fn=collate_episodes)
        
        print(f"\nCreating {args.architecture} model")
        model = ARCHITECTURES[args.architecture]().to(device)
        
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0.01)
        
        print(f"Model created and initialized on {device}")
        
        maml = l2l.algorithms.MAML(
            model, 
            lr=args.inner_lr,
            first_order=False,
            allow_unused=True,
            allow_nograd=True
        )
        
        for param in maml.parameters():
            param.requires_grad = True
        
        optimizer = torch.optim.Adam(maml.parameters(), lr=args.outer_lr)
        scaler = torch.cuda.amp.GradScaler()
        
        print("\nStarting training...")
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            try:
                train_loss, train_acc = train_epoch(
                    maml, train_loader, optimizer, device,
                    args.adaptation_steps, scaler
                )
                
                # Clear memory before validation
                torch.cuda.empty_cache()
                gc.collect()
                
                val_loss, val_acc = validate(
                    maml, val_loader, device,
                    args.adaptation_steps
                )
                
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    # Save checkpoint
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'maml_state_dict': maml.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                    }
                    torch.save(checkpoint, os.path.join(arch_dir, 'best_model.pt'))
                    del checkpoint  # Free memory
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        print(f'Early stopping triggered after {epoch + 1} epochs')
                        break
                
                # Clear memory after each epoch
                torch.cuda.empty_cache()
                gc.collect()
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"WARNING: GPU OOM error in epoch {epoch+1}. Trying to recover...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e
        
        print("\nTesting on individual tasks...")
        checkpoint = torch.load(os.path.join(arch_dir, 'best_model.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        maml.load_state_dict(checkpoint['maml_state_dict'])
        
        test_results = {}
        for task in PB_TASKS:
            print(f"\nTesting on task: {task}")
            test_dataset = SameDifferentDataset(args.data_dir, [task], 'test', support_sizes=[args.support_size])
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                                   num_workers=1, pin_memory=True,
                                   collate_fn=collate_episodes)
            
            # Clear memory before testing each task
            torch.cuda.empty_cache()
            gc.collect()
            
            test_loss, test_acc = validate(
                maml, test_loader, device, 
                args.test_adaptation_steps
            )
            
            test_results[task] = {
                'loss': test_loss,
                'accuracy': test_acc
            }
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        results = {
            'test_results': test_results,
            'best_val_metrics': {
                'loss': checkpoint['val_loss'],
                'accuracy': checkpoint['val_acc'],
                'epoch': checkpoint['epoch']
            },
            'args': vars(args)
        }
        
        with open(os.path.join(arch_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nResults saved to: {arch_dir}")
    
    except Exception as e:
        print(f"\nERROR: Training failed with error: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error details: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main() 