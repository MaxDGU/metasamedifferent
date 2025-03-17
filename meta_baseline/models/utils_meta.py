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
    total_batches = len(val_loader)
    processed_batches = 0
    skipped_batches = 0
    batch_loss = 0.0
    batch_acc = 0.0
    
    task_metrics = {}
    pbar = tqdm(val_loader, desc="Validating")
    
    for batch_idx, batch in enumerate(pbar):
        try:
            # Get batch data and reshape to merge batch and episode dimensions
            support_images = batch['support_images'].to(device)  # [B, N, C, H, W]
            support_labels = batch['support_labels'].to(device)  # [B, N]
            query_images = batch['query_images'].to(device)    # [B, M, C, H, W]
            query_labels = batch['query_labels'].to(device)    # [B, M]
            task = batch['task'][0] if isinstance(batch['task'], list) else batch['task']
            
            # Initialize task metrics if not already present
            if task not in task_metrics:
                task_metrics[task] = {'acc': [], 'loss': [], 'processed': 0, 'skipped': 0}
            
            # Reshape tensors to merge batch and episode dimensions
            B, N, C, H, W = support_images.shape
            support_images = support_images.view(-1, C, H, W)  # [B*N, C, H, W]
            support_labels = support_labels.view(-1)           # [B*N]
            
            _, M, _, _, _ = query_images.shape
            query_images = query_images.view(-1, C, H, W)     # [B*M, C, H, W]
            query_labels = query_labels.view(-1)              # [B*M]
            
            # Clone model for adaptation
            learner = maml.clone()
            layer_lrs = learner.module.get_layer_lrs()
            
            # Adapt on support set
            for _ in range(adaptation_steps):
                with torch.amp.autocast('cuda'):  # Updated to new syntax
                    support_preds = learner(support_images)
                    support_loss = F.binary_cross_entropy_with_logits(
                        support_preds[:, 1], support_labels.float()
                    )
                
                try:
                    grads = torch.autograd.grad(
                        support_loss,
                        learner.parameters(),
                        create_graph=True,
                        allow_unused=True,
                        retain_graph=True
                    )
                    
                    # Update parameters that have gradients
                    for (name, param), grad in zip(learner.named_parameters(), grads):
                        if grad is not None:
                            lr = layer_lrs.get(name, torch.tensor(0.01).to(device))
                            param.data = param.data - lr.abs() * grad
                except RuntimeError as e:
                    if "graph" in str(e):
                        print(f"Warning: Graph error in validation. Skipping step.")
                        continue
                    else:
                        raise e
            
            # Evaluate on query set
            with torch.no_grad():
                with torch.amp.autocast('cuda'):  # Updated to new syntax
                    query_preds = learner(query_images)
                    query_loss = F.binary_cross_entropy_with_logits(
                        query_preds[:, 1], query_labels.float()
                    )
                    query_acc = accuracy(query_preds, query_labels)
            
            batch_loss += query_loss.item()
            batch_acc += query_acc.item()
            processed_batches += 1
            
            # Update task-specific metrics
            task_metrics[task]['acc'].append(query_acc.item())
            task_metrics[task]['loss'].append(query_loss.item())
            task_metrics[task]['processed'] += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': batch_loss / processed_batches,
                'acc': batch_acc / processed_batches,
                'processed': processed_batches,
                'skipped': skipped_batches
            })
            
            # Clear some memory
            del support_images, support_labels, query_images, query_labels
            del learner, layer_lrs, support_preds, query_preds
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\nWarning: GPU OOM error in validation batch {batch_idx}. Task: {task}")
                skipped_batches += 1
                task_metrics[task]['skipped'] += 1
                torch.cuda.empty_cache()
                
                # If we're skipping too many batches, raise an error
                skip_ratio = skipped_batches / (processed_batches + skipped_batches)
                if skip_ratio > 0.25:  # If we're skipping more than 25% of batches
                    raise RuntimeError(
                        f"Skipping too many batches ({skipped_batches}/{processed_batches + skipped_batches}). "
                        "Try reducing batch size or model complexity."
                    )
                continue
            else:
                raise e
    
    # Only calculate averages if we processed any batches
    if processed_batches == 0:
        raise RuntimeError("No batches were processed during validation!")
    
    avg_loss = batch_loss / processed_batches
    avg_acc = batch_acc / processed_batches
    
    # Print detailed metrics
    print("\nValidation Summary:")
    print(f"Processed batches: {processed_batches}/{total_batches}")
    print(f"Skipped batches: {skipped_batches} ({skipped_batches/total_batches*100:.1f}%)")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Average accuracy: {avg_acc:.4f}")
    
    print("\nPer-task validation metrics:")
    for task, metrics in task_metrics.items():
        if metrics['processed'] > 0:
            task_acc = np.mean(metrics['acc'])
            task_loss = np.mean(metrics['loss'])
            print(f"{task}: Acc = {task_acc:.4f}, Loss = {task_loss:.4f}")
            print(f"  Processed: {metrics['processed']}, Skipped: {metrics['skipped']}")
    
    return avg_loss, avg_acc

def collate_episodes(batch):
    """Collate function that combines episodes into a batch dictionary with padding."""
    if not batch:
        return {}
    
    # Initialize dictionary with empty lists for each key
    batch_dict = {
        'support_images': [],
        'support_labels': [],
        'query_images': [],
        'query_labels': [],
        'task': [],
        'support_size': []
    }
    
    # First pass to get maximum sizes
    max_support = max(episode['support_images'].size(0) for episode in batch)
    max_query = max(episode['query_images'].size(0) for episode in batch)
    
    # Second pass to pad and collect
    for episode in batch:
        # Pad support set if needed
        support_images = episode['support_images']
        support_labels = episode['support_labels']
        if support_images.size(0) < max_support:
            pad_size = max_support - support_images.size(0)
            support_images = torch.cat([
                support_images,
                torch.zeros(pad_size, *support_images.shape[1:], device=support_images.device)
            ])
            support_labels = torch.cat([
                support_labels,
                torch.zeros(pad_size, device=support_labels.device)
            ])
        
        # Pad query set if needed
        query_images = episode['query_images']
        query_labels = episode['query_labels']
        if query_images.size(0) < max_query:
            pad_size = max_query - query_images.size(0)
            query_images = torch.cat([
                query_images,
                torch.zeros(pad_size, *query_images.shape[1:], device=query_images.device)
            ])
            query_labels = torch.cat([
                query_labels,
                torch.zeros(pad_size, device=query_labels.device)
            ])
        
        # Add to batch dictionary
        batch_dict['support_images'].append(support_images)
        batch_dict['support_labels'].append(support_labels)
        batch_dict['query_images'].append(query_images)
        batch_dict['query_labels'].append(query_labels)
        batch_dict['task'].append(episode['task'])
        batch_dict['support_size'].append(episode['support_size'])
    
    # Stack tensors
    batch_dict['support_images'] = torch.stack(batch_dict['support_images'])
    batch_dict['support_labels'] = torch.stack(batch_dict['support_labels'])
    batch_dict['query_images'] = torch.stack(batch_dict['query_images'])
    batch_dict['query_labels'] = torch.stack(batch_dict['query_labels'])
    
    # Keep task and support_size as lists
    batch_dict['task'] = batch_dict['task'][0]  # Just take the first task since they should all be the same
    batch_dict['support_size'] = batch_dict['support_size'][0]  # Same for support_size
    
    return batch_dict

def train_epoch(maml, train_loader, optimizer, device, adaptation_steps, scaler):
    """Train for one epoch."""
    maml.train()
    total_batches = len(train_loader)
    batch_loss = 0
    batch_acc = 0
    
    # Debug: Print initial learning rate gradients
    print("\nChecking learning rate parameters:")
    for name, param in maml.module.named_parameters():
        if 'lr_' in name:
            print(f"{name}: requires_grad={param.requires_grad}, grad={param.grad}")
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()
        
        try:
            # Get batch data and reshape to merge batch and episode dimensions
            support_images = batch['support_images'].to(device)  # [B, N, C, H, W]
            support_labels = batch['support_labels'].to(device)  # [B, N]
            query_images = batch['query_images'].to(device)    # [B, M, C, H, W]
            query_labels = batch['query_labels'].to(device)    # [B, M]
            
            # Reshape tensors to merge batch and episode dimensions
            B, N, C, H, W = support_images.shape
            support_images = support_images.view(-1, C, H, W)  # [B*N, C, H, W]
            support_labels = support_labels.view(-1)           # [B*N]
            
            _, M, _, _, _ = query_images.shape
            query_images = query_images.view(-1, C, H, W)     # [B*M, C, H, W]
            query_labels = query_labels.view(-1)              # [B*M]
            
            # Adapt the model on the support set
            learner = maml.clone()
            
            # Debug: Print learning rates being used
            if batch_idx == 0:
                print("\nLearning rates for first batch:")
                layer_lrs = learner.module.get_layer_lrs()
                for name, lr in layer_lrs.items():
                    print(f"{name}: lr={lr.item()}")
            
            for _ in range(adaptation_steps):
                with torch.amp.autocast('cuda'):  # Updated to new syntax
                    support_preds = learner(support_images)
                    support_loss = F.binary_cross_entropy_with_logits(
                        support_preds[:, 1], support_labels.float()
                    )
                
                try:
                    # Compute gradients with allow_unused=True
                    grads = torch.autograd.grad(
                        support_loss,
                        learner.parameters(),
                        create_graph=True,
                        allow_unused=True,
                        retain_graph=True
                    )
                    
                    # Update parameters that have gradients
                    layer_lrs = learner.module.get_layer_lrs()
                    for (name, param), grad in zip(learner.named_parameters(), grads):
                        if grad is not None:  # Only update if gradient exists
                            lr = layer_lrs.get(name, torch.tensor(0.01).to(device))
                            param.data = param.data - lr.abs() * grad
                except RuntimeError as e:
                    if "graph" in str(e):
                        print(f"Warning: Graph error in adaptation. Skipping step.")
                        continue
                    else:
                        raise e
            
            # Evaluate on query set
            with torch.amp.autocast('cuda'):  # Updated to new syntax
                query_preds = learner(query_images)
                query_loss = F.binary_cross_entropy_with_logits(
                    query_preds[:, 1], query_labels.float()
                )
                query_acc = accuracy(query_preds, query_labels)
            
            # Scale loss and compute gradients
            scaled_loss = scaler.scale(query_loss)
            scaled_loss.backward(retain_graph=True)
            
            # Debug: Print learning rate gradients after backward
            if batch_idx == 0:
                print("\nLearning rate gradients after backward:")
                for name, param in maml.module.named_parameters():
                    if 'lr_' in name:
                        print(f"{name}: grad={param.grad}")
            
            # Unscale gradients and check for infs/nans
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(maml.parameters(), 1.0)
            
            # Update weights if gradients are valid
            scaler.step(optimizer)
            scaler.update()
            
            batch_loss += query_loss.item()
            batch_acc += query_acc.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': batch_loss / (batch_idx + 1),
                'acc': batch_acc / (batch_idx + 1)
            })
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"WARNING: GPU OOM error in batch. Skipping...")
                torch.cuda.empty_cache()
                continue
            else:
                raise e
    
    return batch_loss / total_batches, batch_acc / total_batches 