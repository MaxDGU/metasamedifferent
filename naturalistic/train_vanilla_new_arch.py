import os
import sys
# Add parent directory to Python path to find baselines package
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Commenting out

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

# Make sure the meta_baseline directory is in the Python path
# This allows us to import the newer 'lr' models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the newer models from meta_baseline
from meta_baseline.models.conv2lr import SameDifferentCNN as Conv2lrCNN
from meta_baseline.models.conv4lr import SameDifferentCNN as Conv4lrCNN
from meta_baseline.models.conv6lr import SameDifferentCNN as Conv6lrCNN

# Define ARCHITECTURES dictionary to map string names to the correct model classes
ARCHITECTURES = {
    'conv2lr': Conv2lrCNN,
    'conv4lr': Conv4lrCNN,
    'conv6lr': Conv6lrCNN
}

class NaturalisticDataset(Dataset):
    """Dataset for naturalistic same/different classification."""
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: Path to objectsall_2 directory
            split: One of 'train', 'val', or 'test'
            transform: PyTorch transforms to apply.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        
        # Get all image paths and labels
        same_dir = self.root_dir / split / 'same'
        diff_dir = self.root_dir / split / 'different'
        
        if not same_dir.exists() or not diff_dir.exists():
            raise FileNotFoundError(f"Data directory for split '{split}' not found or incomplete in {self.root_dir}")
        
        same_files = list(same_dir.glob('*.png'))
        diff_files = list(diff_dir.glob('*.png'))
        
        self.file_paths = same_files + diff_files
        self.labels = ([1] * len(same_files)) + ([0] * len(diff_files))
        
        # Convert to tensor
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        
        if transform is None:
            # Default transforms if none provided
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet statistics
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        return {'image': image, 'label': label}

def train_epoch(model, train_loader, criterion, optimizer, device, epoch_num, total_epochs):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch_num}/{total_epochs} [Train]')
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        
        running_loss += loss.item() * images.size(0)
        running_acc += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        
        pbar.set_postfix({
            'loss': running_loss / total_samples if total_samples > 0 else 0,
            'acc': running_acc / total_samples if total_samples > 0 else 0
        })
    
    avg_loss = running_loss / total_samples if total_samples > 0 else 0
    avg_acc = running_acc / total_samples if total_samples > 0 else 0
    return avg_loss, avg_acc

def validate(model, val_loader, criterion, device, epoch_num, total_epochs):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0
    
    pbar_desc = f'Epoch {epoch_num}/{total_epochs} [Val]' if epoch_num is not None else 'Validating'
    pbar = tqdm(val_loader, desc=pbar_desc)
    
    with torch.no_grad():
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            
            running_loss += loss.item() * images.size(0)
            running_acc += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            pbar.set_postfix({
                'loss': running_loss / total_samples if total_samples > 0 else 0,
                'acc': running_acc / total_samples if total_samples > 0 else 0
            })

    avg_loss = running_loss / total_samples if total_samples > 0 else 0
    avg_acc = running_acc / total_samples if total_samples > 0 else 0
    return avg_loss, avg_acc

def main(args):
    # Set random seeds
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            # Consider adding these for full reproducibility, though they can slow things down
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
    
    # Create output directory (unique per seed and arch)
    # The SLURM script will pass output_dir as logs_naturalistic_vanilla/arch/seed_X
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Output directory: {output_dir}")
    print(f"Running with arguments: {args}")

    # Define transforms (consistent across splits)
    data_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    try:
        train_dataset = NaturalisticDataset(args.data_dir, 'train', transform=data_transforms)
        val_dataset = NaturalisticDataset(args.data_dir, 'val', transform=data_transforms)
    except FileNotFoundError as e:
        print(f"Error initializing dataset: {e}")
        print("Please ensure your --data_dir contains train/ and val/ subdirectories with same/ and different/ images.")
        return
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True if device.type == 'cuda' else False)
    
    # Create model
    if args.architecture in ARCHITECTURES:
        model = ARCHITECTURES[args.architecture]()
    else:
        raise ValueError(f"Unknown architecture: {args.architecture}. Available: {list(ARCHITECTURES.keys())}")
    
    model.to(device)
    print(f"Created {args.architecture} model.")
    
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"Using Adam optimizer with LR: {args.lr}, Weight Decay: {args.weight_decay}")
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        print(f"Using SGD optimizer with LR: {args.lr}, Momentum: {args.momentum}, Weight Decay: {args.weight_decay}")
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
        
    best_val_acc = 0.0
    patience_counter = 0
    
    metrics = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'test_loss': None, 'test_acc': None, # Test metrics will be filled by a separate test script
        'best_val_epoch': 0,
        'total_epochs_trained': 0,
        'final_lr': args.lr # Storing final LR, could be useful if LR scheduling is added
    }
    
    print(f"Starting training for {args.epochs} epochs with patience {args.patience}.")

    for epoch in range(1, args.epochs + 1):
        metrics['total_epochs_trained'] = epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, args.epochs)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch}/{args.epochs} Results: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            metrics['best_val_epoch'] = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'args': args # Save args for future reference
            }, output_dir / 'best_model.pt')
            print(f"  New best validation accuracy: {best_val_acc:.4f}. Saved model.")
        else:
            patience_counter += 1
            print(f"  Validation accuracy did not improve. Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print("Early stopping triggered!")
                break
    
    print(f"Training finished. Best validation accuracy: {best_val_acc:.4f} at epoch {metrics['best_val_epoch']}.")
    
    # Save final metrics (even if early stopped)
    with open(output_dir / 'training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4, cls=NpEncoder) # Use NpEncoder for numpy types in args

# Custom JSON encoder to handle numpy types that might be in args
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path): # Handle Path objects in args
            return str(obj)
        return super(NpEncoder, self).default(obj)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a vanilla CNN on Naturalistic Same-Different Dataset")
    
    # Model and Data
    parser.add_argument('--architecture', type=str, required=True, choices=list(ARCHITECTURES.keys()), help='Model architecture name')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the root of the naturalistic dataset (e.g., trainsize_6400_1200-300-100)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save logs, checkpoints, and metrics (e.g., logs_vanilla/conv2lr/seed_42)')
    
    # Training Hyperparameters
    parser.add_argument('--optimizer', type=str.lower, default='adam', choices=['adam', 'sgd'], help='Optimizer type (adam or sgd)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 penalty)')
    
    # Training Control
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of DataLoader workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    main(args) 



    '''
    python naturalistic/train_vanilla.py \
    --architecture conv2 \
    --seed 42 \
    --data_dir naturalistic/objectsall_2/aligned/N_16/trainsize_6400_1200-300-100 \
    --output_dir results/naturalistic/conv2/seed_42
    
    '''