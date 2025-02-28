import os
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import learn2learn as l2l
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import math
import random
from conv6lr import SameDifferentCNN, accuracy, EarlyStopping, SameDifferentDataset
import sys

#pb tasks and SVRT, can ablate either from training or testing; used only PB tasks in paper experiments
svrt_tasks = ['1', '7', '5', '15', '16', '19', '20', '21', '22']
pb_tasks = ['regular', 'lines', 'open', 'wider_line', 'scrambled', 
            'random_color', 'arrows', 'irregular', 'filled', 'original']


def create_datasets(data_dir, train_tasks, test_task, support_sizes=[4, 6, 8, 10]):
    """Create train, validation, and test datasets with specified tasks."""
    train_dataset = SameDifferentDataset(data_dir, train_tasks, 'train', support_sizes)
    val_dataset = SameDifferentDataset(data_dir, train_tasks, 'val', support_sizes)
    
    test_dataset = SameDifferentDataset(data_dir, [test_task], 'test', support_sizes)
    
    print(f"\nDataset Split Info:")
    print(f"Training tasks: {train_tasks}")
    print(f"Test task (held-out): {test_task}")
    print(f"Training episodes: {len(train_dataset)}")
    print(f"Validation episodes: {len(val_dataset)}")
    print(f"Test episodes: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset

def validate(maml, val_dataset, device, meta_batch_size=8, num_adaptation_steps=5, max_episodes=200):
    """Validate the model on the validation dataset."""
    maml.module.eval()
    val_loss = 0.0
    val_acc = 0.0
    
    if isinstance(val_dataset, list):
        episodes = val_dataset
        num_batches = 1
        task_metrics = {}
        for episode in episodes:
            task = episode['task']
            if task not in task_metrics:
                task_metrics[task] = {'acc': [], 'loss': []}
    else:
        num_batches = max_episodes // meta_batch_size
        episodes = val_dataset.get_balanced_batch(meta_batch_size)
        task_metrics = {task: {'acc': [], 'loss': []} for task in val_dataset.tasks}
    
    pbar = tqdm(range(num_batches), desc="Validating")
    
    for _ in pbar:
        batch_loss = 0.0
        batch_acc = 0.0
        
        for episode in episodes:
            task = episode['task']
            learner = maml.clone()
            
            support_images = episode['support_images'].to(device)
            support_labels = episode['support_labels'].unsqueeze(1).to(device)
            query_images = episode['query_images'].to(device)
            query_labels = episode['query_labels'].unsqueeze(1).to(device)
            
            # Inner loop adaptation
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
        
        # Calculate running task accuracies
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

def main(held_out_task, seed=None, save_path='model.pt', data_dir=None, return_metrics=False):
    """Main training function that holds out one PB task and trains on remaining PB tasks + SVRT tasks."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed) if torch.cuda.is_available() else None
    
    metrics = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_metrics': {}
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if data_dir is None:
        pb_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'pb', 'pb')
        svrt_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'svrt_fixed')
    else:
        pb_data_dir = os.path.join(data_dir, 'pb', 'pb')
        svrt_data_dir = os.path.join(data_dir, 'svrt_fixed')
    
    # Create train/val/test datasets
    # Include all PB tasks except the held out one
    train_pb_tasks = [task for task in pb_tasks if task != held_out_task]
    
    train_pb_dataset = SameDifferentDataset(
        pb_data_dir,
        train_pb_tasks,
        'train'
    )
    
    train_svrt_dataset = SameDifferentDataset(
        svrt_data_dir,
        svrt_tasks,  # Include all SVRT tasks in training
        'train'
    )
    
    # Create validation datasets
    val_pb_dataset = SameDifferentDataset(
        pb_data_dir,
        train_pb_tasks,
        'val'
    )
    
    val_svrt_dataset = SameDifferentDataset(
        svrt_data_dir,
        svrt_tasks,
        'val'
    )
    
    # Create test dataset for held out PB task
    test_dataset = SameDifferentDataset(
        pb_data_dir,
        [held_out_task],
        'test'
    )
    
    # Configuration
    meta_batch_size = 32
    num_epochs = 100
    adaptation_steps = 5
    gradient_accumulation_steps = 4
    
    # Create model and move to GPU
    model = SameDifferentCNN()
    model = model.to(device)
    
    maml = l2l.algorithms.MAML(model, 
                              lr=None,
                              first_order=True,
                              allow_unused=True, 
                              allow_nograd=True)
    
    opt = torch.optim.Adam(maml.parameters(), 
                          lr=1e-3)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=10)
    scaler = torch.cuda.amp.GradScaler()
    
    # Training loop
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        opt.zero_grad()
        
        # Calculate number of episodes to sample from each dataset
        total_episodes = len(train_pb_dataset) + len(train_svrt_dataset)
        pb_ratio = len(train_pb_dataset) / total_episodes
        svrt_ratio = len(train_svrt_dataset) / total_episodes
        
        pb_episodes_per_batch = max(1, int(meta_batch_size * pb_ratio))
        svrt_episodes_per_batch = meta_batch_size - pb_episodes_per_batch
        
        num_batches = total_episodes // meta_batch_size
        pbar = tqdm(range(num_batches), desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx in pbar:
            batch_losses = []
            batch_acc = 0.0
            
            # Get balanced batch of episodes from both datasets
            pb_episodes = train_pb_dataset.get_balanced_batch(pb_episodes_per_batch)
            svrt_episodes = train_svrt_dataset.get_balanced_batch(svrt_episodes_per_batch)
            episodes = pb_episodes + svrt_episodes
            random.shuffle(episodes)
            
            for episode in episodes:
                learner = maml.clone()
                
                # Move data to GPU
                support_images = episode['support_images'].to(device)
                support_labels = episode['support_labels'].unsqueeze(1).to(device)
                query_images = episode['query_images'].to(device)
                query_labels = episode['query_labels'].unsqueeze(1).to(device)
                
                # Inner loop adaptation with mixed precision
                with torch.cuda.amp.autocast():
                    layer_lrs = learner.module.get_layer_lrs()
                    for _ in range(adaptation_steps):
                        support_preds = learner(support_images)
                        support_loss = F.binary_cross_entropy_with_logits(
                            support_preds[:, 1], support_labels.squeeze(1).float())
                        
                        grads = torch.autograd.grad(support_loss, learner.parameters(),
                                                  create_graph=True, allow_unused=True)
                        
                        for (name, param), grad in zip(learner.named_parameters(), grads):
                            if grad is not None:
                                lr = layer_lrs.get(name, torch.tensor(0.01).to(device))
                                param.data = param.data - lr.abs() * grad
                    
                    # Compute query loss
                    query_preds = learner(query_images)
                    query_loss = F.binary_cross_entropy_with_logits(
                        query_preds[:, 1], query_labels.squeeze(1).float())
                    query_acc = accuracy(query_preds, query_labels)
                
                # Scale loss and accumulate gradients
                scaled_loss = query_loss / (meta_batch_size * gradient_accumulation_steps)
                scaler.scale(scaled_loss).backward()
                
                batch_losses.append(scaled_loss.item())
                batch_acc += query_acc.item()
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(maml.parameters(), max_norm=5.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            
            running_loss += sum(batch_losses)
            running_acc += batch_acc / len(episodes)
            
            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.4f}',
                'acc': f'{running_acc/(batch_idx+1):.4f}'
            })
        
        # Store training metrics
        epoch_loss = running_loss / num_batches
        epoch_acc = running_acc / num_batches
        metrics['train_loss'].append(epoch_loss)
        metrics['train_acc'].append(epoch_acc)
        
        # Validate on training tasks every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            # Get balanced batch of validation episodes from both datasets
            val_episodes = val_pb_dataset.get_balanced_batch(meta_batch_size // 2) + \
                         val_svrt_dataset.get_balanced_batch(meta_batch_size // 2)
            random.shuffle(val_episodes)
            
            # Validate on combined validation set
            val_loss, val_acc = validate(maml, val_episodes, device, meta_batch_size=meta_batch_size)
            print(f'\nEpoch {epoch+1}: Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}')
            
            # Store validation metrics
            metrics['val_loss'].append(val_loss)
            metrics['val_acc'].append(val_acc)
            
            # Early stopping check
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
                    'optimizer_state_dict': opt.state_dict(),
                    'val_acc': best_val_acc,
                }, save_path)
    
    # Final test on held-out task
    print(f"\nTesting on held-out task: {held_out_task}")
    test_adaptation_steps = 15  # Increased adaptation steps for testing
    print(f"Using {test_adaptation_steps} adaptation steps for testing")
    test_loss, test_acc = validate(maml, test_dataset, device, 
                                 meta_batch_size=meta_batch_size,
                                 num_adaptation_steps=test_adaptation_steps)
    print(f'Results on held-out task {held_out_task}:')
    print(f'Loss = {test_loss:.4f}, Accuracy = {test_acc:.4f}')
    
    metrics['test_metrics'][held_out_task] = {
        'loss': test_loss,
        'acc': test_acc,
        'num_adaptation_steps': test_adaptation_steps
    }
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'train_tasks': {'pb': train_pb_tasks, 'svrt': svrt_tasks},
        'held_out_task': held_out_task,
        'val_acc': best_val_acc,
        'test_acc': test_acc,
    }, save_path.replace('.pt', '_final.pt'))
    
    if return_metrics:
        return metrics

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python conv6lr_holdout.py <held_out_task>")
        print(f"Available tasks: {pb_tasks}")
        sys.exit(1)
    
    held_out_task = sys.argv[1]
    if held_out_task not in pb_tasks:
        print(f"Error: {held_out_task} is not a valid PB task")
        print(f"Available tasks: {pb_tasks}")
        sys.exit(1)
    
    main(held_out_task) 