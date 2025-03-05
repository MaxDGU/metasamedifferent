import os
import json
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
from tqdm import tqdm
import random
import argparse
from PIL import Image
from models.conv2 import SameDifferentCNN as Conv2CNN
from models.conv4 import SameDifferentCNN as Conv4CNN
from models.conv6 import SameDifferentCNN as Conv6CNN
from models.utils import SameDifferentDataset, train_epoch, validate, EarlyStopping

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_task', type=str, required=True, help='PB task to test on')
    parser.add_argument('--architecture', type=str, required=True, choices=['conv2', 'conv4', 'conv6'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--val_freq', type=int, default=10)
    parser.add_argument('--improvement_threshold', type=float, default=0.02)
    parser.add_argument('--data_dir', type=str, default='data/pb/pb')
    parser.add_argument('--output_dir', type=str, default='results/pb_baselines')
    args = parser.parse_args()
    
    # Define all PB tasks
    PB_TASKS = ['regular', 'lines', 'open', 'wider_line', 'scrambled', 
                'random_color', 'arrows', 'irregular', 'filled', 'original']
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory - new structure reflecting training on all tasks
    model_dir = os.path.join(args.output_dir, 'all_tasks', args.architecture, f'test_{args.test_task}', f'seed_{args.seed}')
    os.makedirs(model_dir, exist_ok=True)
    
    # Create datasets and dataloaders
    # Training on all tasks
    train_dataset = SameDifferentDataset(args.data_dir, PB_TASKS, split='train')
    # Validation and testing on specified task only
    val_dataset = SameDifferentDataset(args.data_dir, args.test_task, split='val')
    test_dataset = SameDifferentDataset(args.data_dir, args.test_task, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    if args.architecture == 'conv2':
        model = Conv2CNN()
    elif args.architecture == 'conv4':
        model = Conv4CNN()
    else:  # conv6
        model = Conv6CNN()
    
    model = model.to(device)
    
    # Setup training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_val_acc = 0
    training_history = []
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.improvement_threshold)
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Only validate every val_freq epochs
        if (epoch + 1) % args.val_freq == 0:
            # Validate
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            # Save history
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            })
            
            # Check if validation accuracy improved
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }, os.path.join(model_dir, 'best_model.pt'))
                print(f'New best validation accuracy: {val_acc*100:.2f}%')
            
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%')
            print(f'Best Val Acc: {best_val_acc*100:.2f}%')
            
            # Early stopping check
            early_stopping(val_acc)
            if early_stopping.should_stop:
                print(f'Early stopping triggered after epoch {epoch+1}')
                print(f'No improvement of {args.improvement_threshold*100}% or more in validation accuracy for {args.patience} validations')
                break
        else:
            # Just save training metrics
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': None,
                'val_acc': None
            })
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%')
    
    # Load best model and evaluate on test set
    if os.path.exists(os.path.join(model_dir, 'best_model.pt')):
        checkpoint = torch.load(os.path.join(model_dir, 'best_model.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    else:
        print("No best model found, using final model state")
    
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f'\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%')
    
    # Save metrics
    metrics = {
        'args': vars(args),
        'training_history': training_history,
        'best_val_acc': best_val_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'total_epochs': epoch + 1
    }
    
    with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == '__main__':
    main() 