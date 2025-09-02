import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from PIL import Image
import numpy as np
from tqdm import tqdm
import random
import json
import argparse
from pathlib import Path
from torchvision import transforms

# Import models from baselines package
from baselines.models.conv2 import SameDifferentCNN as Conv2CNN
from baselines.models.conv4 import SameDifferentCNN as Conv4CNN
from baselines.models.conv6 import SameDifferentCNN as Conv6CNN

class NaturalisticDataset(Dataset):
    """Dataset for naturalistic same/different classification."""
    
    def __init__(self, root_dir, split='train'):
        """
        Args:
            root_dir: Path to N_16 or N_32 directory
            split: One of 'train', 'val', or 'test'
        """
        self.root_dir = Path(root_dir)
        self.split = split
        
        # Get all image paths and labels
        same_dir = self.root_dir / split / 'same'
        diff_dir = self.root_dir / split / 'different'
        
        same_files = list(same_dir.glob('*.png'))
        diff_files = list(diff_dir.glob('*.png'))
        
        self.file_paths = same_files + diff_files
        self.labels = ([1] * len(same_files)) + ([0] * len(diff_files))
        
        # Convert to tensor
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        
        # Define transforms
        # Note: Using ImageNet normalization for naturalistic images
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet statistics
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        return {
            'image': image,
            'label': label
        }

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        running_loss += loss.item() * labels.size(0)
        running_acc += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss/total,
            'acc': running_acc/total
        })
    
    return running_loss/total, running_acc/total

def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            running_loss += loss.item() * labels.size(0)
            running_acc += (predicted == labels).sum().item()
    
    return running_loss/total, running_acc/total

def main(args):
    # Set random seeds
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed) if torch.cuda.is_available() else None
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = NaturalisticDataset(args.data_dir, 'train')
    val_dataset = NaturalisticDataset(args.data_dir, 'val')
    test_dataset = NaturalisticDataset(args.data_dir, 'test')
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Create model
    if args.architecture == 'conv2':
        model = Conv2CNN()
    elif args.architecture == 'conv4':
        model = Conv4CNN()
    elif args.architecture == 'conv6':
        model = Conv6CNN()
    else:
        raise ValueError(f"Unknown architecture: {args.architecture}")
    
    model = model.to(device)
    print(f"Created {args.architecture} model")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    
    metrics = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_loss': None,
        'test_acc': None
    }
    
    for epoch in range(100):  # Max 100 epochs
        print(f"\nEpoch {epoch+1}/100")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, os.path.join(args.output_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break
    
    # Load best model and evaluate on test set
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    metrics['test_loss'] = test_loss
    metrics['test_acc'] = test_acc
    
    print(f"\nFinal Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    
    # Save results
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str, required=True, choices=['conv2', 'conv4', 'conv6'])
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to N_16 or N_32 directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loading workers')
    args = parser.parse_args() 
    
    main(args) 



    '''
    python train_vanilla_cluster.py \
    --architecture conv2 \
    --seed 42 \
    --data_dir data/naturalistic/N_16/trainsize_6400_1200-300-100 \
    --output_dir results/naturalistic/conv2/seed_42
    --batch_size 4
    --num_workers 1
    '''