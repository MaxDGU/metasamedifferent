"""
Shared utilities for meta-learning baseline models.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
import json
import h5py
from torch.utils.data import Dataset, DataLoader
import glob
from PIL import Image
from torchvision import transforms

class SameDifferentDataset(Dataset):
    def __init__(self, data_dir, tasks, split, support_sizes=[4, 6, 8, 10]):
        """
        Dataset for loading same-different task data.
        
        Args:
            data_dir (str): Path to data directory
                - For PB data: 'data/pb/pb'
                - For SVRT data: 'data/svrt_fixed'
            tasks (list): List of tasks to load
                - For PB: ['regular', 'lines', 'open', etc.]
                - For SVRT: ['1', '7', '5', etc.]
            split (str): One of ['train', 'val', 'test']
            support_sizes (list): List of support set sizes to use
        """
        self.data_dir = data_dir
        self.tasks = tasks
        self.split = split
        self.support_sizes = support_sizes
        
        # Create a list of all possible episode files
        self.episode_files = []
        for task in tasks:
            for support_size in support_sizes:
                # Handle both PB and SVRT data paths
                if 'svrt_fixed' in data_dir:
                    # SVRT test data path
                    file_path = os.path.join(data_dir, 
                                           f'results_problem_{task}',
                                           f'support{support_size}_{split}.h5')
                else:
                    # PB training/validation data path
                    file_path = os.path.join(data_dir,
                                           f'{task}_support{support_size}_{split}.h5')
                
                if os.path.exists(file_path):
                    self.episode_files.append({
                        'file_path': file_path,
                        'task': task,
                        'support_size': support_size
                    })
        
        if not self.episode_files:
            raise ValueError(f"No valid files found for tasks {tasks} in {data_dir}")
        
        # Calculate total number of episodes
        self.total_episodes = 0
        self.file_episode_counts = []
        for file_info in self.episode_files:
            with h5py.File(file_info['file_path'], 'r') as f:
                num_episodes = f['support_images'].shape[0]
                self.file_episode_counts.append(num_episodes)
                self.total_episodes += num_episodes
        
        # Track episodes per task for balanced sampling
        self.task_indices = {task: [] for task in tasks}
        total_idx = 0
        for i, file_info in enumerate(self.episode_files):
            task = file_info['task']
            num_episodes = self.file_episode_counts[i]
            self.task_indices[task].extend(
                range(total_idx, total_idx + num_episodes))
            total_idx += num_episodes
        
        # Debug prints
        print(f"\nDataset initialization for {split} split:")
        print(f"Found {len(self.episode_files)} valid files")
        print(f"Total episodes: {self.total_episodes}")
        for task in tasks:
            print(f"Task {task}: {len(self.task_indices[task])} episodes")
    
    def __len__(self):
        return self.total_episodes
    
    def __getitem__(self, idx):
        # Find which file contains this index
        file_idx = 0
        while idx >= self.file_episode_counts[file_idx]:
            idx -= self.file_episode_counts[file_idx]
            file_idx += 1
        
        file_info = self.episode_files[file_idx]
        
        with h5py.File(file_info['file_path'], 'r') as f:
            support_images = torch.from_numpy(f['support_images'][idx]).float() / 255.0
            support_labels = torch.from_numpy(f['support_labels'][idx]).long()
            query_images = torch.from_numpy(f['query_images'][idx]).float() / 255.0
            query_labels = torch.from_numpy(f['query_labels'][idx]).long()
        
        # Convert from NHWC to NCHW format
        support_images = support_images.permute(0, 3, 1, 2)
        query_images = query_images.permute(0, 3, 1, 2)
        
        return {
            'support_images': support_images,
            'support_labels': support_labels,
            'query_images': query_images,
            'query_labels': query_labels,
            'task': file_info['task'],
            'support_size': file_info['support_size']
        }
    
    def get_balanced_batch(self, batch_size):
        """Get a batch with equal representation from each task"""
        episodes = []
        
        # Filter out tasks with no episodes
        available_tasks = [task for task in self.tasks if self.task_indices[task]]
        
        if not available_tasks:
            raise ValueError(f"No episodes available for any tasks in {self.split} split")
        
        tasks_per_batch = max(1, batch_size // len(available_tasks))
        
        # First, get equal episodes from each available task
        for task in available_tasks:
            # Make sure we don't request more episodes than available
            available_episodes = len(self.task_indices[task])
            n_episodes = min(tasks_per_batch, available_episodes)
            if n_episodes > 0:
                task_episodes = random.sample(self.task_indices[task], n_episodes)
                episodes.extend([self[idx] for idx in task_episodes])
        
        # If we still need more episodes to reach batch_size, sample randomly
        while len(episodes) < batch_size:
            task = random.choice(available_tasks)
            idx = random.choice(self.task_indices[task])
            episodes.append(self[idx])
        
        return episodes

def accuracy(predictions, targets):
    """Calculate binary classification accuracy."""
    with torch.no_grad():
        # Convert 2D logits to binary prediction
        predictions = F.softmax(predictions, dim=1)
        predictions = (predictions[:, 1] > 0.5).float()
        
        # Safely handle targets of different dimensions
        if targets.dim() > 1:
            # If targets has more than 1 dimension, squeeze it to match predictions
            targets = targets.squeeze()
            
            # If after squeezing it still has more than 1 dimension,
            # take the second column (same as predictions)
            if targets.dim() > 1 and targets.shape[1] > 1:
                targets = targets[:, 1]
        
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

def validate(maml, val_loader, device, adaptation_steps=5, inner_lr=None):
    """Validation with learned per-layer learning rates"""
    maml.module.eval()
    val_loss = 0.0
    val_acc = 0.0
    
    num_tasks = len(val_loader.dataset.tasks)
    episodes_per_task = 200 // num_tasks  # Using 200 as default max_episodes
    num_batches = len(val_loader)
    
    task_metrics = {task: {'acc': [], 'loss': []} for task in val_loader.dataset.tasks}
    pbar = tqdm(val_loader, desc="Validating")
    
    for batch in pbar:
        batch_loss = 0.0
        batch_acc = 0.0
        
        for episode in batch:
            task = episode['task']
            learner = maml.clone()
            
            support_images = episode['support_images'].to(device)
            support_labels = episode['support_labels'].to(device)
            query_images = episode['query_images'].to(device)
            query_labels = episode['query_labels'].to(device)
            
            layer_lrs = learner.module.get_layer_lrs()
            with torch.cuda.amp.autocast():
                for _ in range(adaptation_steps):
                    support_preds = learner(support_images)
                    support_loss = F.binary_cross_entropy_with_logits(
                        support_preds[:, 1], support_labels.float())
                    
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
                        query_preds[:, 1], query_labels.float())
                    query_acc = accuracy(query_preds, query_labels)
            
            batch_loss += query_loss.item()
            batch_acc += query_acc.item()
            
            task_metrics[task]['acc'].append(query_acc.item())
            task_metrics[task]['loss'].append(query_loss.item())
        
        batch_loss /= len(batch)
        batch_acc /= len(batch)
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

def collate_episodes(batch):
    """Collate function that preserves episodes as a list"""
    return batch

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
                
                with torch.cuda.amp.autocast():
                    # Inner loop adaptation
                    for _ in range(adaptation_steps):
                        support_preds = learner(support_images)
                        support_loss = F.binary_cross_entropy_with_logits(
                            support_preds[:, 1], support_labels.float())
                        learner.adapt(support_loss, allow_unused=True, allow_nograd=True)
                    
                    # Evaluate on query set
                    query_preds = learner(query_images)
                    query_loss = F.binary_cross_entropy_with_logits(
                        query_preds[:, 1], query_labels.float())
                    query_acc = accuracy(query_preds, query_labels)
                
                scaled_loss = scaler.scale(query_loss)
                scaled_loss.backward(retain_graph=True)  # Add retain_graph=True
                
                batch_loss += query_loss.item()
                batch_acc += query_acc.item()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"WARNING: GPU OOM error in batch. Skipping...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        scaler.step(optimizer)
        scaler.update()
        
        batch_loss /= len(batch)
        batch_acc /= len(batch)
        total_loss += batch_loss
        total_acc += batch_acc
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{batch_loss:.4f}',
            'acc': f'{batch_acc:.4f}'
        })
    
    return total_loss / num_batches, total_acc / num_batches 