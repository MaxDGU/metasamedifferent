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
import random
import glob
from conv6lr import SameDifferentDataset
import json
import argparse
import gc
import sys

train_tasks =  ['regular', 'lines', 'open', 'wider_line', 'scrambled',
                 'random_color', 'arrows', 'irregular', 'filled', 'original']

class SameDifferentCNN(nn.Module):
    def __init__(self):
        super(SameDifferentCNN, self).__init__()
        
        # 4-layer CNN from Kim et al.
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32, track_running_stats=False)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64, track_running_stats=False)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128, track_running_stats=False)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256, track_running_stats=False)
        
        self.pool = nn.MaxPool2d(2)
        self.dropout2d = nn.Dropout2d(0.3)
        
        self._to_linear = None
        self._initialize_size()
        
        # FC layers with decreasing sizes
        self.fc_layers = nn.ModuleList([
            nn.Linear(self._to_linear, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(512),
            nn.LayerNorm(256),
            nn.LayerNorm(128)
        ])
        
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(3)
        ])
        
        self.classifier = nn.Linear(128, 2)
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Learnable per-layer learning rates: initialized to 0.01
        self.lr_conv = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.01) for _ in range(4)
        ])
        self.lr_fc = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.01) for _ in range(3)
        ])
        self.lr_classifier = nn.Parameter(torch.ones(1) * 0.01)
        
        self._initialize_weights()
    
    def _initialize_size(self):
        x = torch.randn(1, 3, 128, 128)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.reshape(x.size(0), -1)
        self._to_linear = x.size(1)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.Linear) and m != self.classifier:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.01)
        
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout2d(x)
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2d(x)
        
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout2d(x)
        
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout2d(x)
        
        x = x.reshape(x.size(0), -1)
        
        for fc, ln, dropout in zip(self.fc_layers, self.layer_norms, self.dropouts):
            x = dropout(F.relu(ln(fc(x))))
        
        x = self.classifier(x)
        return F.softmax(x / self.temperature.abs(), dim=1)
    
    def get_layer_lrs(self):
        """Return a dictionary mapping parameters to their learning rates"""
        lrs = {}
        
        for i, (conv, bn) in enumerate(zip(
            [self.conv1, self.conv2, self.conv3, self.conv4],
            [self.bn1, self.bn2, self.bn3, self.bn4]
        )):
            lrs.update({name: self.lr_conv[i].abs() for name, _ in conv.named_parameters()})
            lrs.update({name: self.lr_conv[i].abs() for name, _ in bn.named_parameters()})
        
        for i, (fc, ln) in enumerate(zip(self.fc_layers, self.layer_norms)):
            lrs.update({name: self.lr_fc[i].abs() for name, _ in fc.named_parameters()})
            lrs.update({name: self.lr_fc[i].abs() for name, _ in ln.named_parameters()})
        
        lrs.update({name: self.lr_classifier.abs() for name, _ in self.classifier.named_parameters()})
        
        return lrs

def accuracy(predictions, targets):
    """Calculate binary classification accuracy"""
    with torch.no_grad():
        predictions = F.softmax(predictions, dim=1)
        predictions = (predictions[:, 1] > 0.5).float()
        targets = targets.squeeze(1)
        return (predictions == targets).float().mean()

def load_model(model_path, model, optimizer=None):
    """Load model and optimizer state from checkpoint"""
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint

class EarlyStopping:
    """Early stopping with patience and minimum improvement threshold"""
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_val = None
        self.should_stop = False
    
    def __call__(self, val_acc):
        if self.best_val is None:
            self.best_val = val_acc
        elif val_acc < self.best_val + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_val = val_acc   
            self.counter = 0

def validate(maml, val_dataset, device, meta_batch_size=8, num_adaptation_steps=5, max_episodes=200):
    """Validation with learned per-layer learning rates"""
    maml.module.eval()
    val_loss = 0.0
    val_acc = 0.0
    
    num_tasks = len(val_dataset.tasks)
    episodes_per_task = max_episodes // num_tasks
    num_batches = max_episodes // meta_batch_size
    
    task_metrics = {task: {'acc': [], 'loss': []} for task in val_dataset.tasks}
    pbar = tqdm(range(num_batches), desc="Validating")
    
    for _ in pbar:
        batch_loss = 0.0
        batch_acc = 0.0
        
        episodes = val_dataset.get_balanced_batch(meta_batch_size)
        
        for episode in episodes:
            task = episode['task']
            learner = maml.clone()
            
            support_images = episode['support_images'].to(device)
            support_labels = episode['support_labels'].unsqueeze(1).to(device)
            query_images = episode['query_images'].to(device)
            query_labels = episode['query_labels'].unsqueeze(1).to(device)
            
            layer_lrs = learner.module.get_layer_lrs()
            for _ in range(num_adaptation_steps):
                support_preds = learner(support_images)
                support_loss = F.binary_cross_entropy_with_logits(
                    support_preds[:, 1], support_labels.squeeze(1).float())
                
                grads = torch.autograd.grad(support_loss, learner.parameters(),
                                          create_graph=True,
                                          allow_unused=True)
                
                for (name, param), grad in zip(learner.named_parameters(), grads):
                    if grad is not None:
                        lr = layer_lrs.get(name, torch.tensor(0.01).to(device))
                        param.data = param.data - lr.abs() * grad
            
            with torch.no_grad():
                query_preds = learner(query_images)
                query_loss = F.binary_cross_entropy_with_logits(
                    query_preds[:, 1], query_labels.squeeze(1).float())
                query_acc = accuracy(query_preds, query_labels)
            
            batch_loss += query_loss.item()
            batch_acc += query_acc.item()
            
            task_metrics[task]['acc'].append(query_acc.item())
            task_metrics[task]['loss'].append(query_loss.item())
        
        batch_loss /= len(episodes)
        batch_acc /= len(episodes)
        val_loss += batch_loss
        val_acc += batch_acc
        
        task_accs = {task: np.mean(metrics['acc']) if metrics['acc'] else 0.0
                    for task, metrics in task_metrics.items()}
        
        pbar.set_postfix({
            'loss': f'{batch_loss:.4f}',
            'acc': f'{batch_acc:.4f}',
            'task_accs': {t: f'{acc:.2f}' for t, acc in task_accs.items()}
        })
    
    print("\nValidation Results by Task:")
    for task in task_metrics:
        task_acc = np.mean(task_metrics[task]['acc']) if task_metrics[task]['acc'] else 0.0
        task_loss = np.mean(task_metrics[task]['loss']) if task_metrics[task]['loss'] else 0.0
        print(f"{task}: Acc = {task_acc:.4f}, Loss = {task_loss:.4f}")
    
    return val_loss / num_batches, val_acc / num_batches

def train_epoch(maml, train_loader, optimizer, device, adaptation_steps, scaler):
    """Train using MAML with learned per-layer learning rates"""
    maml.train()
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        batch_loss = 0
        batch_acc = 0
        optimizer.zero_grad()
        
        for episode in batch:
            try:
                learner = maml.clone()
                
                support_images = episode['support_images'].to(device, non_blocking=True)
                support_labels = episode['support_labels'].to(device, non_blocking=True)
                query_images = episode['query_images'].to(device, non_blocking=True)
                query_labels = episode['query_labels'].to(device, non_blocking=True)
                
                for _ in range(adaptation_steps):
                    support_preds = learner(support_images)
                    support_loss = F.binary_cross_entropy_with_logits(
                        support_preds[:, 1], support_labels.float())
                    learner.adapt(support_loss)
                
                with torch.cuda.amp.autocast():
                    query_preds = learner(query_images)
                    query_loss = F.binary_cross_entropy_with_logits(
                        query_preds[:, 1], query_labels.float())
                    query_acc = accuracy(query_preds, query_labels)
                
                scaled_loss = scaler.scale(query_loss)
                scaled_loss.backward()
                
                batch_loss += query_loss.item()
                batch_acc += query_acc.item()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"WARNING: GPU OOM error in batch. Trying to recover...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                    continue
                else:
                    raise e
        
        batch_loss /= len(batch)
        batch_acc /= len(batch)
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(maml.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += batch_loss
        total_acc += batch_acc
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{batch_loss:.4f}',
            'acc': f'{batch_acc:.4f}'
        })
    
    return total_loss / num_batches, total_acc / num_batches

def collate_episodes(batch):
    """Collate function that preserves episodes as a list"""
    return batch

#EXPERIMENT 1
def main(seed=None, output_dir=None, pb_data_dir='data/pb/pb'):
    """Main training function with support for resuming from checkpoint"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/pb/pb',
                        help='Directory containing the PB dataset')
    parser.add_argument('--output_dir', type=str, default='results/meta_baselines',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, required=True,
                        help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=32,
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
    args = parser.parse_args()
    
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This script requires GPU access.")
        device = torch.device('cuda')
        print(f"Using device: {device}")
        
        if not os.path.exists(args.data_dir):
            raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
        
        if args.seed is not None:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)
            torch.cuda.manual_seed(args.seed)
        
        arch_dir = os.path.join(args.output_dir, 'conv4', f'seed_{args.seed}')
        os.makedirs(arch_dir, exist_ok=True)
        
        print("\nCreating datasets...")
        train_dataset = SameDifferentDataset(args.data_dir, train_tasks, 'train', support_sizes=[args.support_size])
        val_dataset = SameDifferentDataset(args.data_dir, train_tasks, 'val', support_sizes=[args.support_size])
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                num_workers=4, pin_memory=True, collate_fn=collate_episodes)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                              num_workers=4, pin_memory=True, collate_fn=collate_episodes)
        
        print("\nCreating conv4 model")
        model = SameDifferentCNN().to(device)
        
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
        patience = 10
        patience_counter = 0
        
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            try:
                train_loss, train_acc = train_epoch(
                    maml, train_loader, optimizer, device,
                    args.adaptation_steps, scaler
                )
                
                val_loss, val_acc = validate(
                    maml, val_loader, device,
                    args.adaptation_steps, args.inner_lr
                )
                
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'maml_state_dict': maml.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                    }, os.path.join(arch_dir, 'best_model.pt'))
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f'Early stopping triggered after {epoch + 1} epochs')
                        break
            
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
        for task in train_tasks:
            print(f"\nTesting on task: {task}")
            test_dataset = SameDifferentDataset(args.data_dir, [task], 'test', support_sizes=[args.support_size])
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                                   num_workers=4, pin_memory=True, collate_fn=collate_episodes)
            
            test_loss, test_acc = validate(
                maml, test_loader, device,
                args.test_adaptation_steps, args.inner_lr
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
        sys.exit(1)

if __name__ == '__main__':
    main()


