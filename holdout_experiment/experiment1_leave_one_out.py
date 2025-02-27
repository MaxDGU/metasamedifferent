#!/bin/env python
import os
import torch
import torch.nn.functional as F
from conv6lr import SameDifferentCNN, SameDifferentDataset, accuracy, collate_episodes
import learn2learn as l2l
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import random
import json
from datetime import datetime
import argparse
import copy
import gc

def train_epoch(maml, train_loader, optimizer, device, adaptation_steps, scaler):
    """Single training epoch with improved monitoring"""
    maml.train()
    total_loss = 0
    total_acc = 0
    n_batches = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, episodes in enumerate(pbar):
        optimizer.zero_grad()
        batch_loss = 0
        batch_acc = 0
        
        # Process each episode with mixed precision
        with torch.cuda.amp.autocast():
            for episode in episodes:
                learner = maml.clone()
                
                # Move data to GPU
                support_images = episode['support_images'].to(device, non_blocking=True)
                support_labels = episode['support_labels'].unsqueeze(1).to(device, non_blocking=True)
                query_images = episode['query_images'].to(device, non_blocking=True)
                query_labels = episode['query_labels'].unsqueeze(1).to(device, non_blocking=True)
                
                # Inner loop adaptation with loss checking
                for step in range(adaptation_steps):
                    support_preds = learner(support_images)
                    support_loss = F.binary_cross_entropy_with_logits(
                        support_preds[:, 1], support_labels.squeeze(1).float())
                    
                    # Check for abnormal loss values
                    if support_loss.item() > 10 and step == 0:
                        print(f"\nWARNING: High support loss: {support_loss.item():.4f}")
                        print("Support predictions:", torch.sigmoid(support_preds[:, 1]).detach().cpu().numpy())
                        print("Support labels:", support_labels.squeeze(1).cpu().numpy())
                    
                    grads = torch.autograd.grad(
                        support_loss,
                        learner.parameters(),
                        create_graph=True,
                        allow_unused=True
                    )
                    
                    # Gradient norm clipping for stability
                    grad_norm = torch.norm(torch.stack([torch.norm(g) for g in grads if g is not None]))
                    if grad_norm > 10:
                        scaling_factor = 10 / grad_norm
                        grads = [g * scaling_factor if g is not None else None for g in grads]
                    
                    for param, grad in zip(learner.parameters(), grads):
                        if grad is not None:
                            param.data = param.data - maml.lr * grad
                
                # Query loss and accuracy
                query_preds = learner(query_images)
                query_loss = F.binary_cross_entropy_with_logits(
                    query_preds[:, 1], query_labels.squeeze(1).float())
                
                # Check for abnormal query loss
                if query_loss.item() > 10:
                    print(f"\nWARNING: High query loss: {query_loss.item():.4f}")
                    print("Query predictions:", torch.sigmoid(query_preds[:, 1]).detach().cpu().numpy())
                    print("Query labels:", query_labels.squeeze(1).cpu().numpy())
                
                query_acc = accuracy(query_preds, query_labels)
                
                batch_loss += query_loss
                batch_acc += query_acc.item()
        
        # Scale loss and backward pass
        scaled_loss = batch_loss / len(episodes)
        
        # Additional loss value check
        if scaled_loss.item() > 10:
            print(f"\nWARNING: High batch loss: {scaled_loss.item():.4f}")
        
        scaler.scale(scaled_loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(maml.parameters(), max_norm=10.0)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        # Update metrics
        total_loss += scaled_loss.item()
        total_acc += batch_acc / len(episodes)
        n_batches += 1
        
        # Update progress bar with more information
        pbar.set_postfix({
            'loss': f'{total_loss/n_batches:.4f}',
            'acc': f'{total_acc/n_batches:.4f}',
            'batch_loss': f'{scaled_loss.item():.4f}'
        })
        
        # Early warning for unstable training
        if batch_idx == 0 and total_loss/n_batches > 10:
            print("\nWARNING: Initial loss is very high. Consider:")
            print("1. Reducing learning rates (current inner_lr={}, outer_lr={})".format(
                maml.lr, optimizer.param_groups[0]['lr']))
            print("2. Checking data normalization")
            print("3. Adjusting model initialization")
    
    return total_loss / n_batches, total_acc / n_batches

def validate(model, val_dataloader, criterion, device, adaptation_steps, inner_lr):
    model.eval()
    total_val_loss = 0
    total_val_acc = 0
    num_batches = 0

    for batch_idx, episodes in enumerate(val_dataloader):
        batch_loss = 0
        batch_acc = 0
        
        for episode in episodes:
            # Move data to GPU and handle dimensions
            support_images = episode['support_images'].to(device, non_blocking=True)
            support_labels = episode['support_labels'].unsqueeze(1).to(device, non_blocking=True)
            query_images = episode['query_images'].to(device, non_blocking=True)
            query_labels = episode['query_labels'].unsqueeze(1).to(device, non_blocking=True)

            # Clone the model for adaptation
            adapted_model = copy.deepcopy(model)
            adapted_model.train()  # Put in training mode for adaptation
            
            # Ensure parameters require gradients for adaptation
            for param in adapted_model.parameters():
                param.requires_grad_(True)

            # Support set adaptation
            for step in range(adaptation_steps):
                support_preds = adapted_model(support_images)
                support_loss = criterion(
                    support_preds[:, 1],  # Use second logit for binary classification
                    support_labels.squeeze(1).float()
                )
                
                # Compute gradients for inner loop
                grads = torch.autograd.grad(
                    support_loss,
                    adapted_model.parameters(),
                    create_graph=True,
                    retain_graph=True
                )
                
                # Manual parameter update
                for param, grad in zip(adapted_model.parameters(), grads):
                    param.data = param.data - inner_lr * grad

            # Evaluate on query set
            adapted_model.eval()
            with torch.no_grad():
                query_preds = adapted_model(query_images)
                query_loss = criterion(
                    query_preds[:, 1],  # Use second logit for binary classification
                    query_labels.squeeze(1).float()
                )
                
                # Calculate accuracy using the same method as training
                query_acc = accuracy(query_preds, query_labels)
                
                batch_loss += query_loss.item()
                batch_acc += query_acc.item()
        
        # Average over episodes in the batch
        avg_batch_loss = batch_loss / len(episodes)
        avg_batch_acc = batch_acc / len(episodes)
        
        total_val_loss += avg_batch_loss
        total_val_acc += avg_batch_acc
        num_batches += 1

    avg_val_loss = total_val_loss / num_batches
    avg_val_acc = total_val_acc / num_batches

    return avg_val_loss, avg_val_acc

def test_model(model, test_loader, device, test_adaptation_steps, inner_lr):
    """More robust testing function with better error handling and monitoring"""
    try:
        model.eval()
        total_loss = 0
        total_acc = 0
        num_batches = 0
        
        print(f"\nTesting with {test_adaptation_steps} adaptation steps")
        print(f"Dataset size: {len(test_loader.dataset)} episodes")
        
        # Use tqdm for progress tracking
        for episodes in tqdm(test_loader, desc="Testing episodes"):
            batch_loss = 0
            batch_acc = 0
            
            for episode in episodes:
                try:
                    # Clone model for this episode
                    adapted_model = copy.deepcopy(model)
                    adapted_model.train()
                    
                    # Move data to GPU
                    support_images = episode['support_images'].to(device, non_blocking=True)
                    support_labels = episode['support_labels'].unsqueeze(1).to(device, non_blocking=True)
                    query_images = episode['query_images'].to(device, non_blocking=True)
                    query_labels = episode['query_labels'].unsqueeze(1).to(device, non_blocking=True)
                    
                    # Support set adaptation
                    for _ in range(test_adaptation_steps):
                        support_preds = adapted_model(support_images)
                        support_loss = F.binary_cross_entropy_with_logits(
                            support_preds[:, 1], support_labels.squeeze(1).float())
                        
                        grads = torch.autograd.grad(
                            support_loss,
                            adapted_model.parameters(),
                            create_graph=True,
                            allow_unused=True
                        )
                        
                        # Manual parameter update
                        for param, grad in zip(adapted_model.parameters(), grads):
                            if grad is not None:
                                param.data = param.data - inner_lr * grad
                    
                    # Evaluate on query set
                    adapted_model.eval()
                    with torch.no_grad():
                        query_preds = adapted_model(query_images)
                        query_loss = F.binary_cross_entropy_with_logits(
                            query_preds[:, 1], query_labels.squeeze(1).float())
                        query_acc = accuracy(query_preds, query_labels)
                        
                        batch_loss += query_loss.item()
                        batch_acc += query_acc.item()
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"WARNING: GPU OOM error. Trying to recover...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()
                        continue
                    else:
                        raise e
            
            # Average over episodes
            avg_batch_loss = batch_loss / len(episodes)
            avg_batch_acc = batch_acc / len(episodes)
            
            total_loss += avg_batch_loss
            total_acc += avg_batch_acc
            num_batches += 1
            
            # Clear GPU memory after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Calculate final metrics
        final_loss = total_loss / num_batches
        final_acc = total_acc / num_batches
        
        print(f"Test Loss: {final_loss:.4f}")
        print(f"Test Accuracy: {final_acc:.4f}")
        
        return final_loss, final_acc
    
    except Exception as e:
        print(f"Error in testing: {str(e)}")
        return None, None

def main(args):
    # Set random seeds
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory with seed
    args.output_dir = os.path.join(args.output_dir, f'seed_{args.seed}')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device and enable optimizations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Define all PB tasks and remove held out task
    all_tasks = ['regular', 'lines', 'open', 'wider_line', 'scrambled', 
                 'random_color', 'arrows', 'irregular', 'filled', 'original']
    train_tasks = [task for task in all_tasks if task != args.held_out_task]
    print(f"\nTraining on tasks: {train_tasks}")
    print(f"Held out task for testing: {args.held_out_task}")
    print(f"Using seed: {args.seed}")
    
    # Create datasets
    train_dataset = SameDifferentDataset(args.pb_data_dir, train_tasks, 'train')
    val_dataset = SameDifferentDataset(args.pb_data_dir, train_tasks, 'val')
    test_dataset = SameDifferentDataset(args.pb_data_dir, [args.held_out_task], 'test')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_episodes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_episodes
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_episodes
    )
    
    # Create model and move to GPU
    model = SameDifferentCNN().to(device)
    
    # Create MAML wrapper with fixed learning rate
    maml = l2l.algorithms.MAML(
        model,
        lr=args.inner_lr,
        first_order=False,  # Use second-order gradients
        allow_unused=True,
        allow_nograd=True
    )
    
    # Create optimizer and gradient scaler
    optimizer = torch.optim.Adam(maml.parameters(), lr=args.outer_lr)
    scaler = torch.cuda.amp.GradScaler()
    
    # Initialize metrics tracking
    best_val_acc = 0
    best_epoch = 0
    val_history = []  # Track validation accuracies for early stopping
    
    metrics = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_metrics': None,
        'best_epoch': None,
        'seed': args.seed
    }
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Training phase
        train_loss, train_acc = train_epoch(
            maml, train_loader, optimizer, device,
            args.adaptation_steps, scaler
        )
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        
        # Validation phase every 10 epochs
        if (epoch + 1) % 10 == 0:
            print("\nRunning validation...")
            val_loss, val_acc = validate(
                model, val_loader, F.binary_cross_entropy_with_logits, device,
                args.adaptation_steps, args.inner_lr
            )
            metrics['val_loss'].append(val_loss)
            metrics['val_acc'].append(val_acc)
            val_history.append(val_acc)
            
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'metrics': metrics,
                    'seed': args.seed
                }, os.path.join(args.output_dir, 'best_model.pt'))
            
            # Early stopping check - less than 2% improvement over last 3 validations
            if len(val_history) >= 3:
                max_recent = max(val_history[-3:])
                min_recent = min(val_history[-3:])
                improvement = max_recent - min_recent
                
                # Stop if validation accuracy reaches 99% or higher
                if val_acc >= 0.99:
                    print(f"\nEarly stopping! Reached {val_acc*100:.2f}% validation accuracy.")
                    break
                # Or stop if improvement is less than 2%
                elif improvement < 0.02:
                    print(f"\nEarly stopping! Less than 2% improvement in last 3 validations.")
                    print(f"Recent validation accuracies: {val_history[-3:]}")
                    break
        
        # Regular checkpointing
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'seed': args.seed
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt'))
    
    # Load best model for testing
    print("\nLoading best model for testing...")
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test on held out task
    print(f"\nTesting on held out task: {args.held_out_task}")
    print(f"Using {args.test_adaptation_steps} adaptation steps for testing (vs {args.adaptation_steps} for training)")
    
    test_loss, test_acc = test_model(
        model, test_loader, device,
        args.test_adaptation_steps, args.inner_lr
    )
    
    if test_loss is not None and test_acc is not None:
        metrics['test_metrics'] = {
            'loss': test_loss,
            'accuracy': test_acc,
            'adaptation_steps': args.test_adaptation_steps
        }
    else:
        print("WARNING: Testing failed, no test metrics recorded")
        metrics['test_metrics'] = None
    
    metrics['best_epoch'] = best_epoch
    
    # Save final results
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("\nFinal Results:")
    print(f"Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
    if test_loss is not None and test_acc is not None:
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pb_data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--held_out_task', type=str, required=True,
                        choices=['regular', 'lines', 'open', 'wider_line', 'scrambled',
                                'random_color', 'arrows', 'irregular', 'filled', 'original'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--inner_lr', type=float, default=0.01)
    parser.add_argument('--outer_lr', type=float, default=0.0001)
    parser.add_argument('--adaptation_steps', type=int, default=5,
                        help='Number of adaptation steps during training')
    parser.add_argument('--test_adaptation_steps', type=int, default=15,
                        help='Number of adaptation steps during testing (can be higher than training)')
    parser.add_argument('--seed', type=int, required=True,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    main(args) 