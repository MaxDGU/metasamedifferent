"""
Shared utilities for baseline models.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import glob
from PIL import Image
from torchvision import transforms
import json
from torch.utils.data import DataLoader
import tqdm
import h5py


ARCHITECTURES = ['conv2', 'conv4', 'conv6']
SEEDS = [47, 48, 49, 50, 51] 
PB_TASKS = ['regular', 'lines', 'open', 'wider_line', 'scrambled', 'random_color', 
            'arrows', 'irregular', 'filled', 'original'] 

class SameDifferentDataset(Dataset):
    """Dataset for PB tasks with balanced task representation."""
    def __init__(self, data_dir, task_names, split='train', support_sizes=[4, 6, 8, 10], transform=None):
        self.data_dir = data_dir
        self.task_names = task_names if isinstance(task_names, list) else [task_names]
        self.split = split
        self.support_sizes = support_sizes
        self.transform = transform
        
        self.task_data = {task: {'images': [], 'labels': []} for task in self.task_names}
        #tasks are defined in a list from 
        for task_name in self.task_names:
            #support sizes go from 4 to 10 in steps of 2
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
            
            self.task_data[task_name]['images'] = np.array(self.task_data[task_name]['images'])
            self.task_data[task_name]['labels'] = np.array(self.task_data[task_name]['labels'])
            
            print(f"Loaded {len(self.task_data[task_name]['images'])} images for task {task_name}")
            print(f"Label distribution: {np.bincount(self.task_data[task_name]['labels'].astype(int))}")
        
        self.total_size = sum(len(data['images']) for data in self.task_data.values())
        self.samples_per_task = min(len(data['images']) for data in self.task_data.values())
        
        self.task_indices = {
            task: np.arange(len(data['images'])) 
            for task, data in self.task_data.items()
        }
        
        for indices in self.task_indices.values():
            np.random.shuffle(indices)
        
        self.current_pos = {task: 0 for task in self.task_names}
    
    def __len__(self):
        #all tasks should have equal representation 
        return self.samples_per_task * len(self.task_names)
    
    def __getitem__(self, idx):
        #lookup for task index 
        task_idx = idx % len(self.task_names)
        task_name = self.task_names[task_idx]
        
        pos = self.current_pos[task_name]
        if pos >= len(self.task_indices[task_name]):
            #reshuffle indices if we've gone through all samples
            np.random.shuffle(self.task_indices[task_name])
            self.current_pos[task_name] = 0
            pos = 0
        
        actual_idx = self.task_indices[task_name][pos]
        self.current_pos[task_name] += 1
        
        image = self.task_data[task_name]['images'][actual_idx]
        label = self.task_data[task_name]['labels'][actual_idx]
        
        # Convert to float and normalize to [-1, 1]
        image = torch.FloatTensor(image.transpose(2, 0, 1)) / 127.5 - 1.0
        label = torch.tensor(int(label))
        
        return {'image': image, 'label': label, 'task': task_name}

def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = F.cross_entropy(outputs, labels.long())
        loss.backward()
        
        optimizer.step()
        preds = outputs.argmax(dim=1)
        acc = (preds == labels).float().mean()
        
        running_loss += loss.item()
        running_acc += acc.item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/len(train_loader):.4f}',
            'acc': f'{running_acc/len(train_loader):.4f}'
        })
    
    return running_loss / len(train_loader), running_acc / len(train_loader)

def validate(model, val_loader, device):
    """Validate the model."""
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels.long())
            
            preds = outputs.argmax(dim=1)
            acc = (preds == labels).float().mean()
            
            val_loss += loss.item()
            val_acc += acc.item()
    
    return val_loss / len(val_loader), val_acc / len(val_loader)

class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_val = None
        self.should_stop = False
    
    def __call__(self, acc):
        if self.best_val is None:
            self.best_val = acc
        elif acc < self.best_val + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_val = acc
            self.counter = 0

def train_model(model_class, args):
    """Generic training function for baseline models."""
    #can specify seeds in advance for tracking
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed) if torch.cuda.is_available() else None
    
    #determine device type, if running on cluster this will be cuda, else cpu for testing 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    num_epochs = 100
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create datasets and dataloaders
    train_dataset = args.dataset_class(args.data_dir, args.task, 'train')
    val_dataset = args.dataset_class(args.data_dir, args.task, 'test')  # Using test split as validation
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    model = model_class().to(device)
    print(f"Model created on {device}")
    
    #lr is outer loop learning rate here 
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01
    )
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=20)
    best_val_acc = 0.0
    train_accs = []
    val_accs = []
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        #import train_epoch from utils and validate 
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        train_accs.append(train_acc)
        
        val_loss, val_acc = validate(model, val_loader, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        val_accs.append(val_acc)
        
        # Early stopping check on validation accuracy
        early_stopping(val_acc)
        if early_stopping.should_stop:
            print("Early stopping triggered!")
            break
        
        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc,
            }, os.path.join(args.output_dir, 'best_model.pt'))
    
    # Load best model for final evaluation
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final validation pass (using this as test set)
    final_val_loss, final_val_acc = validate(model, val_loader, device)
    print(f"\nFinal Results:")
    print(f"Test Loss: {final_val_loss:.4f}")
    print(f"Test Accuracy: {final_val_acc:.4f}")
    
    # Save results in the format expected by run_baselines.py
    results = {
        'train_acc': train_accs,
        'val_acc': val_accs,
        'test_acc': final_val_acc
    }
    
    results_file = os.path.join(args.output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_file}") 