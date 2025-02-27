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

class PBDataset(Dataset):
    """Dataset for PB tasks with balanced task representation."""
    def __init__(self, data_dir, task_names, split='train', support_sizes=[4, 6, 8, 10], transform=None):
        self.data_dir = data_dir
        self.task_names = task_names if isinstance(task_names, list) else [task_names]
        self.split = split
        self.support_sizes = support_sizes
        self.transform = transform
        
        # Dictionary to store data per task
        self.task_data = {task: {'images': [], 'labels': []} for task in self.task_names}
        
        # Process each task separately
        for task_name in self.task_names:
            # Process each support size
            for support_size in support_sizes:
                filename = f"{task_name}_support{support_size}_{split}.h5"
                filepath = os.path.join(data_dir, filename)
                
                if not os.path.exists(filepath):
                    print(f"Warning: File not found: {filepath}")
                    continue
                
                print(f"Loading {filename}")
                with h5py.File(filepath, 'r') as f:
                    num_episodes = f['support_images'].shape[0]
                    
                    for episode_idx in range(num_episodes):
                        support_images = f['support_images'][episode_idx]
                        support_labels = f['support_labels'][episode_idx]
                        query_images = f['query_images'][episode_idx]
                        query_labels = f['query_labels'][episode_idx]
                        
                        all_images = np.concatenate([support_images, query_images])
                        all_labels = np.concatenate([support_labels, query_labels])
                        
                        self.task_data[task_name]['images'].extend(all_images)
                        self.task_data[task_name]['labels'].extend(all_labels)
            
            # Convert to numpy arrays
            self.task_data[task_name]['images'] = np.array(self.task_data[task_name]['images'])
            self.task_data[task_name]['labels'] = np.array(self.task_data[task_name]['labels'])
            
            print(f"Loaded {len(self.task_data[task_name]['images'])} images for task {task_name}")
            print(f"Label distribution: {np.bincount(self.task_data[task_name]['labels'].astype(int))}")
        
        # Calculate total size and samples per task per batch
        self.total_size = sum(len(data['images']) for data in self.task_data.values())
        self.samples_per_task = min(len(data['images']) for data in self.task_data.values())
        
        # Create indices for balanced sampling
        self.task_indices = {
            task: np.arange(len(data['images'])) 
            for task, data in self.task_data.items()
        }
        
        # Shuffle indices for each task
        for indices in self.task_indices.values():
            np.random.shuffle(indices)
        
        # Keep track of current position in each task
        self.current_pos = {task: 0 for task in self.task_names}
    
    def __len__(self):
        # Return the size that ensures all tasks are seen equally
        return self.samples_per_task * len(self.task_names)
    
    def __getitem__(self, idx):
        # Determine which task to sample from
        task_idx = idx % len(self.task_names)
        task_name = self.task_names[task_idx]
        
        # Get current position for this task
        pos = self.current_pos[task_name]
        if pos >= len(self.task_indices[task_name]):
            # Reshuffle indices if we've gone through all samples
            np.random.shuffle(self.task_indices[task_name])
            self.current_pos[task_name] = 0
            pos = 0
        
        # Get the actual index from our shuffled indices
        actual_idx = self.task_indices[task_name][pos]
        self.current_pos[task_name] += 1
        
        # Get the image and label
        image = self.task_data[task_name]['images'][actual_idx]
        label = self.task_data[task_name]['labels'][actual_idx]
        
        # Convert to float and normalize to [-1, 1]
        image = torch.FloatTensor(image.transpose(2, 0, 1)) / 127.5 - 1.0
        label = torch.tensor(int(label))
        
        return {'image': image, 'label': label, 'task': task_name}

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc='Training')
    for batch in progress_bar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device).float()
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs[:, 1], labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        probs = torch.sigmoid(outputs[:, 1])
        predicted = (probs >= 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar with overall metrics
        progress_bar.set_postfix({
            'loss': f'{running_loss/total:.3f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss/len(train_loader), correct/total

def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation')
        for batch in progress_bar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device).float()  # Convert to float for BCE loss
            
            outputs = model(images)
            
            # Use second column of outputs for binary prediction
            loss = criterion(outputs[:, 1], labels)
            
            running_loss += loss.item()
            # Calculate accuracy using probabilities
            probs = torch.sigmoid(outputs[:, 1])  # Convert logits to probabilities
            predicted = (probs >= 0.5).float()  # Use 0.5 threshold on probabilities
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({
                'loss': f'{running_loss/total:.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    return running_loss/len(val_loader), correct/total

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
    train_dataset = PBDataset(args.data_dir, PB_TASKS, split='train')
    # Validation and testing on specified task only
    val_dataset = PBDataset(args.data_dir, args.test_task, split='val')
    test_dataset = PBDataset(args.data_dir, args.test_task, split='test')
    
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
    patience_counter = 0
    training_history = []
    
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
            
            # Check if validation accuracy improved by at least improvement_threshold
            if val_acc > (best_val_acc + args.improvement_threshold):
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }, os.path.join(model_dir, 'best_model.pt'))
                print(f'Validation accuracy improved by more than {args.improvement_threshold*100}%!')
            else:
                patience_counter += 1
                print(f'No significant improvement. Patience: {patience_counter}/{args.patience}')
            
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%')
            print(f'Best Val Acc: {best_val_acc*100:.2f}%')
            
            # Early stopping
            if patience_counter >= args.patience:
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