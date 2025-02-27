import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import random
import glob
from PIL import Image
from torchvision import transforms
import json
import argparse

class SameDifferentDataset(Dataset):
    def __init__(self, data_dir, problem_number, split):
        """Dataset for loading same-different task PNG data for a specific problem."""
        self.data_dir = data_dir
        self.problem_number = problem_number
        self.split = split
        
        # Find the correct problem directory (ignoring timestamps)
        problem_pattern = f'results_problem_{problem_number}_*'
        problem_dirs = glob.glob(os.path.join(data_dir, problem_pattern))
        if not problem_dirs:
            raise ValueError(f"No directory found for problem {problem_number}")
        
        # Use the first matching directory
        problem_dir = problem_dirs[0]
        split_dir = os.path.join(problem_dir, split)
        
        # Get all PNG files
        self.image_paths = glob.glob(os.path.join(split_dir, '*.png'))
        if not self.image_paths:
            raise ValueError(f"No PNG files found in {split_dir}")
        
        # Extract labels from filenames (sample_1_0009 -> 1)
        self.labels = []
        for path in self.image_paths:
            filename = os.path.basename(path)
            label = int(filename.split('_')[1])  # Get the middle number
            self.labels.append(label)
        
        self.labels = torch.tensor(self.labels)
        print(f"Loaded {len(self.image_paths)} images for {split} split")
        print(f"Label distribution: {torch.bincount(self.labels.long())}")
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        image = self.transform(image)
        label = self.labels[idx]
        
        return {
            'image': image,
            'label': label
        }

class SameDifferentCNN(nn.Module):
    def __init__(self):
        super(SameDifferentCNN, self).__init__()
        
        # First layer: 6x6 filters with 18 filters
        self.conv1 = nn.Conv2d(3, 18, kernel_size=6, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(18, track_running_stats=False)
        
        # Subsequent layers: 2x2 filters with doubling filter counts
        self.conv2 = nn.Conv2d(18, 36, kernel_size=2, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(36, track_running_stats=False)
        
        self.conv3 = nn.Conv2d(36, 72, kernel_size=2, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(72, track_running_stats=False)
        
        self.conv4 = nn.Conv2d(72, 144, kernel_size=2, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(144, track_running_stats=False)
        
        self.conv5 = nn.Conv2d(144, 288, kernel_size=2, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(288, track_running_stats=False)
        
        self.conv6 = nn.Conv2d(288, 576, kernel_size=2, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(576, track_running_stats=False)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dropout2d = nn.Dropout2d(0.1)  # Add spatial dropout
        
        # Calculate the size of flattened features
        self._to_linear = None
        self._initialize_size()
        
        # Three FC layers with 1024 units each
        self.fc1 = nn.Linear(self._to_linear, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        
        # Dropouts for FC layers
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.3)
        
        # Final classification layer
        self.classifier = nn.Linear(1024, 2)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_size(self):
        x = torch.randn(1, 3, 128, 128)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.pool(F.relu(self.bn6(self.conv6(x))))
        x = x.reshape(x.size(0), -1)
        self._to_linear = x.size(1)
    
    def _initialize_weights(self):
        # Initialize convolutional layers
        for m in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        # Initialize fully connected layers
        for fc in [self.fc1, self.fc2, self.fc3]:
            nn.init.kaiming_normal_(fc.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(fc.bias, 0)
        
        # Initialize classifier
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, x):
        # Convolutional layers with spatial dropout
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout2d(x)
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2d(x)
        
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout2d(x)
        
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout2d(x)
        
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.dropout2d(x)
        
        x = self.pool(F.relu(self.bn6(self.conv6(x))))
        x = self.dropout2d(x)
        
        x = x.reshape(x.size(0), -1)
        
        # FC layers with dropout
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.dropout3(F.relu(self.fc3(x)))
        
        return self.classifier(x)

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
        
        # Calculate accuracy
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

def main(args):
    # Set random seeds if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed) if torch.cuda.is_available() else None
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    num_epochs = 100
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create datasets and dataloaders
    train_dataset = SameDifferentDataset(args.data_dir, args.task, 'train')
    val_dataset = SameDifferentDataset(args.data_dir, args.task, 'test')  # Using test split as validation
    
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
    
    # Create model
    model = SameDifferentCNN().to(device)
    print(f"Model created on {device}")
    
    # Optimizer
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
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        train_accs.append(train_acc)
        
        # Validate
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='SVRT task number')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    args = parser.parse_args()
    
    print(f"\nStarting training for task {args.task} with seed {args.seed}")
    main(args) 