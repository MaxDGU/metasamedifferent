import os
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import learn2learn as l2l
from learn2learn.data import MetaDataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import math

class SameDifferentDataset(Dataset):
    def __init__(self, data_dir, tasks, mode='train'):
        self.data_dir = data_dir
        self.tasks = tasks
        self.mode = mode
        self.episodes = []
        
        # Load episodes from each task
        for task in tasks:
            file_path = os.path.join(data_dir, f'{task}_{mode}.h5')
            with h5py.File(file_path, 'r') as f:
                num_episodes = f['support_images'].shape[0]
                for i in range(num_episodes):
                    # Convert from NHWC to NCHW format
                    support_images = torch.FloatTensor(f['support_images'][i]).permute(0, 3, 1, 2)
                    query_images = torch.FloatTensor(f['query_images'][i]).permute(0, 3, 1, 2)
                    
                    episode = {
                        'support_images': support_images,
                        'support_labels': torch.FloatTensor(f['support_labels'][i]),
                        'query_images': query_images,
                        'query_labels': torch.FloatTensor(f['query_labels'][i]),
                        'task': task
                    }
                    self.episodes.append(episode)
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        episode = self.episodes[idx]
        return {
            'support_images': (episode['support_images'] / 127.5) - 1.0,
            'support_labels': episode['support_labels'],
            'query_images': (episode['query_images'] / 127.5) - 1.0,
            'query_labels': episode['query_labels'],
            'task': episode['task']
        }

def debug_gradients(model, name=""):
    print(f"\nGradient check for {name}:")
    for name, param in model.named_parameters():
        print(f"{name}:")
        print(f"- requires_grad: {param.requires_grad}")
        print(f"- has grad_fn: {param.grad_fn is not None}")
        print(f"- grad: {param.grad is not None}")
        if hasattr(param, 'grad_fn'):
            print(f"- grad_fn type: {type(param.grad_fn)}")

class SameDifferentCNN(nn.Module):
    def __init__(self):
        #TODO: ensure same architecture as Kim et al 2018
        # 6 convolutional layers with increasing channels:
        # 3 → 18 → 36 → 72 → 144 → 288 → 576
        # MaxPool2d reduces spatial dimensions by 2x after each conv
        # Input: 128x128 → 64x64 → 32x32 → 16x16 → 8x8 → 4x4 → 2x2
        
        # 4 fully connected layers:
        # flattened → 1024 → 1024 → 1024 → 1
        super(SameDifferentCNN, self).__init__()
        
        # Convolutional layers with batch norm
        self.conv1 = nn.Conv2d(3, 18, kernel_size=6, stride=1, padding=3)
        self.bn1 = nn.BatchNorm2d(18, track_running_stats=False)  # No running stats for meta-learning
        
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
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the size of flattened features
        self._to_linear = None
        self._initialize_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 1)
        
        # Initialize weights
        self._initialize_weights()
        
        # Wrap parameters to ensure they require gradients
        for param in self.parameters():
            if param.requires_grad:
                param.data = param.data.clone().detach().requires_grad_(True)
    
    def _initialize_size(self):
        x = torch.randn(1, 3, 128, 128)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 64x64
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 32x32
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 16x16
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # 8x8
        x = self.pool(F.relu(self.bn5(self.conv5(x))))  # 4x4
        x = self.pool(F.relu(self.bn6(self.conv6(x))))  # 2x2
        x = x.reshape(x.size(0), -1)  # Flatten
        self._to_linear = x.size(1)  # Get the flattened dimension size
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Larger initial weights for better gradients
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                # Initialize final layer carefully
                if m == self.fc4:  # Output layer
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                    nn.init.zeros_(m.bias)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(m.bias, 0.1)  # Small positive bias
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.pool(F.relu(self.bn6(self.conv6(x))))
        
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def accuracy(predictions, targets):
    with torch.no_grad():
        predictions = torch.sigmoid(predictions)
        predictions = (predictions > 0.5).float()
        return (predictions == targets).float().mean()

def load_model(model_path, model, optimizer=None):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint

def test_gradient_flow():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SameDifferentCNN().to(device)
    
    # Get a sample input
    dataset = SameDifferentDataset('data/meta_h5', ['regular'], mode='train')
    episode = dataset[0]
    
    # Move data to device
    images = episode['support_images'].to(device)
    labels = episode['support_labels'].unsqueeze(1).to(device)
    
    # Zero gradients
    model.zero_grad()
    
    # Forward pass
    outputs = model(images)
    loss = F.binary_cross_entropy_with_logits(outputs, labels)
    
    # Backward pass
    loss.backward()
    
    # Check gradients layer by layer
    has_gradients = False
    print("\nChecking gradients:")
    
    # Check conv layers
    for i in range(1, 7):
        conv = getattr(model, f'conv{i}')
        if conv.weight.grad is not None and conv.weight.grad.abs().sum() > 0:
            print(f"Conv{i} has gradients")
            has_gradients = True
    
    # Check fc layers
    for i in range(1, 5):
        fc = getattr(model, f'fc{i}')
        if fc.weight.grad is not None and fc.weight.grad.abs().sum() > 0:
            print(f"FC{i} has gradients")
            has_gradients = True
    
    if not has_gradients:
        print("No gradients detected in any layer")
    
    return has_gradients

def create_datasets(data_dir, all_tasks, val_split=0.2):
    """
    Create train, validation, and test datasets with both task and episode splits.
    """
    # Split tasks
    num_tasks = len(all_tasks)
    num_val_tasks = max(1, int(num_tasks * 0.2))  # 20% of tasks for validation
    train_tasks = all_tasks[:-num_val_tasks-1]  # Leave one task for testing
    val_tasks = all_tasks[-num_val_tasks-1:-1]
    test_tasks = [all_tasks[-1]]  # Last task for testing
    
    print(f"Train tasks: {train_tasks}")
    print(f"Validation tasks: {val_tasks}")
    print(f"Test tasks: {test_tasks}")
    
    # Create datasets with episode splits for training tasks
    train_dataset = SameDifferentDataset(data_dir, train_tasks, mode='train')
    val_dataset_seen = SameDifferentDataset(data_dir, train_tasks, mode='test')  # Use test episodes for validation
    val_dataset_unseen = SameDifferentDataset(data_dir, val_tasks, mode='test')
    test_dataset = SameDifferentDataset(data_dir, test_tasks, mode='test')
    
    return train_dataset, val_dataset_seen, val_dataset_unseen, test_dataset

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def validate(maml, val_dataset, device, meta_batch_size=8, num_adaptation_steps=5, max_episodes=50):
    """Run validation on a dataset with limited episodes."""
    maml.module.eval()
    val_loss = 0.0
    val_acc = 0.0
    
    # Limit number of episodes for quick validation
    total_episodes = min(len(val_dataset), max_episodes)
    num_batches = max(1, total_episodes // meta_batch_size)
    
    print(f"\nValidation Info:")
    print(f"Using {total_episodes} episodes out of {len(val_dataset)}")
    
    # Use tqdm for progress tracking
    pbar = tqdm(range(0, num_batches * meta_batch_size, meta_batch_size), 
               desc="Validating")
    
    for batch_idx in pbar:
        batch_loss = 0.0
        batch_acc = 0.0
        valid_tasks = 0
        
        for task_idx in range(meta_batch_size):
            if batch_idx + task_idx >= total_episodes:  # Use total_episodes instead of len(val_dataset)
                continue
                
            valid_tasks += 1
            episode = val_dataset[batch_idx + task_idx]
            learner = maml.clone()
            
            # Rest of validation logic remains the same
            support_images = episode['support_images'].to(device)
            support_labels = episode['support_labels'].unsqueeze(1).to(device)
            query_images = episode['query_images'].to(device)
            query_labels = episode['query_labels'].unsqueeze(1).to(device)
            
            # Adapt using support set
            for _ in range(num_adaptation_steps):
                support_preds = learner(support_images)
                support_loss = F.binary_cross_entropy_with_logits(support_preds, support_labels)
                learner.adapt(support_loss)
            
            # Evaluate on query set
            with torch.no_grad():
                query_preds = learner(query_images)
                query_loss = F.binary_cross_entropy_with_logits(query_preds, query_labels)
                query_acc = accuracy(query_preds, query_labels)
            
            batch_loss += query_loss.item()
            batch_acc += query_acc.item()
        
        if valid_tasks > 0:
            batch_loss /= valid_tasks
            batch_acc /= valid_tasks
            val_loss += batch_loss
            val_acc += batch_acc
            
            pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'acc': f'{batch_acc:.4f}'
            })
    
    return val_loss / num_batches, val_acc / num_batches

def train_epoch(maml, train_dataset, optimizer, scheduler, device, meta_batch_size, max_batches_per_epoch, num_adaptation_steps=5):
    """Run one training epoch."""
    maml.module.train()
    total_loss = 0.0
    total_acc = 0.0
    
    num_batches = min(max_batches_per_epoch, len(train_dataset) // meta_batch_size)
    pbar = tqdm(range(0, num_batches * meta_batch_size, meta_batch_size), 
                desc='Training')
    
    for batch_idx in pbar:
        optimizer.zero_grad()
        batch_loss = 0.0
        batch_acc = 0.0
        
        for task_idx in range(meta_batch_size):
            if batch_idx + task_idx >= len(train_dataset):
                continue
            
            episode = train_dataset[batch_idx + task_idx]
            learner = maml.clone()
            
            # Move data to device
            support_images = episode['support_images'].to(device)
            support_labels = episode['support_labels'].unsqueeze(1).to(device)
            query_images = episode['query_images'].to(device)
            query_labels = episode['query_labels'].unsqueeze(1).to(device)
            
            # Inner loop adaptation
            for _ in range(num_adaptation_steps):
                support_preds = learner(support_images)
                support_loss = F.binary_cross_entropy_with_logits(support_preds, support_labels)
                learner.adapt(support_loss)
            
            # Evaluate on query set
            query_preds = learner(query_images)
            query_loss = F.binary_cross_entropy_with_logits(query_preds, query_labels)
            query_acc = accuracy(query_preds, query_labels)
            
            batch_loss += query_loss
            batch_acc += query_acc.item()
        
        # Average loss and optimize
        batch_loss = batch_loss / meta_batch_size
        batch_loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{batch_loss.item():.4f}',
            'acc': f'{(batch_acc/meta_batch_size):.4f}'
        })
        
        total_loss += batch_loss.item()
        total_acc += batch_acc / meta_batch_size
    
    return total_loss / num_batches, total_acc / num_batches

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    meta_batch_size = 8
    num_epochs = 6
    max_batches_per_epoch = 20
    train_adaptation_steps = 5  # Use 5 steps during training
    val_adaptation_steps = 10   # Use 10 steps during validation
    test_adaptation_steps = [10, 15, 20]  # Test with more steps
    
    # Create datasets
    all_tasks = ['regular', 'lines', 'open', 'wider_line', 'scrambled', 
                 'random_color', 'arrows', 'irregular', 'filled', 'original']
    train_dataset, val_dataset_seen, val_dataset_unseen, test_dataset = create_datasets(
        'data/meta_h5', all_tasks)
    
    # Create model and MAML
    model = SameDifferentCNN()
    model.to(device)
    
    maml = l2l.algorithms.MAML(model, lr=0.001, first_order=True,
                              allow_unused=True, allow_nograd=True)
    
    opt = torch.optim.Adam(maml.parameters(), lr=0.0001)
    
    # Create scheduler
    num_training_steps = num_epochs * (len(train_dataset) // meta_batch_size)
    num_warmup_steps = num_training_steps // 10  # 10% of training for warmup
    scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps, num_training_steps)
    
    # Training loop with validation
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        # Training with 5 adaptation steps
        train_loss, train_acc = train_epoch(maml, train_dataset, opt, scheduler, 
                                          device, meta_batch_size, max_batches_per_epoch,
                                          num_adaptation_steps=train_adaptation_steps)
        
        # Validation with 10 adaptation steps
        val_loss_seen, val_acc_seen = validate(maml, val_dataset_seen, device,
                                             num_adaptation_steps=val_adaptation_steps)
        val_loss_unseen, val_acc_unseen = validate(maml, val_dataset_unseen, device,
                                                  num_adaptation_steps=val_adaptation_steps)
        
        print(f'\nEpoch {epoch+1}:')
        print(f'Train Loss = {train_loss:.4f}, Accuracy = {train_acc:.4f}')
        print(f'Val (Seen) Loss = {val_loss_seen:.4f}, Accuracy = {val_acc_seen:.4f}')
        print(f'Val (Unseen) Loss = {val_loss_unseen:.4f}, Accuracy = {val_acc_unseen:.4f}')
        
        # Save best model
        if val_acc_unseen > best_val_acc:
            best_val_acc = val_acc_unseen
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'val_acc': best_val_acc,
            }, 'best_model.pt')
    
    # Final testing with multiple adaptation steps
    print("\nFinal Testing...")
    for n_steps in test_adaptation_steps:
        test_loss, test_acc = validate(maml, test_dataset, device, 
                                     num_adaptation_steps=n_steps)
        print(f'Test Results ({n_steps} adaptation steps):')
        print(f'Loss = {test_loss:.4f}, Accuracy = {test_acc:.4f}')

if __name__ == '__main__':
    if test_gradient_flow():
        main()
    else:
        print("Gradient flow test failed. Please fix before running full training.") 