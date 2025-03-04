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
    maml.train()
    total_loss = 0
    total_acc = 0
    n_batches = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, episodes in enumerate(pbar):
        optimizer.zero_grad()
        batch_loss = 0
        batch_acc = 0
        
        with torch.cuda.amp.autocast():
            for episode in episodes:
                learner = maml.clone()
                
                support_images = episode['support_images'].to(device, non_blocking=True)
                support_labels = episode['support_labels'].unsqueeze(1).to(device, non_blocking=True)
                query_images = episode['query_images'].to(device, non_blocking=True)
                query_labels = episode['query_labels'].unsqueeze(1).to(device, non_blocking=True)
                
                for step in range(adaptation_steps):
                    support_preds = learner(support_images)
                    support_loss = F.binary_cross_entropy_with_logits(
                        support_preds[:, 1], support_labels.squeeze(1).float())
                    
                    grads = torch.autograd.grad(
                        support_loss,
                        learner.parameters(),
                        create_graph=True,
                        allow_unused=True
                    )
                    
                    grad_norm = torch.norm(torch.stack([torch.norm(g) for g in grads if g is not None]))
                    if grad_norm > 10:
                        scaling_factor = 10 / grad_norm
                        grads = [g * scaling_factor if g is not None else None for g in grads]
                    
                    for param, grad in zip(learner.parameters(), grads):
                        if grad is not None:
                            param.data = param.data - maml.lr * grad
                
                query_preds = learner(query_images)
                query_loss = F.binary_cross_entropy_with_logits(
                    query_preds[:, 1], query_labels.squeeze(1).float())
                query_acc = accuracy(query_preds, query_labels)
                
                batch_loss += query_loss
                batch_acc += query_acc.item()
        
        scaled_loss = batch_loss / len(episodes)
        scaler.scale(scaled_loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(maml.parameters(), max_norm=10.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += scaled_loss.item()
        total_acc += batch_acc / len(episodes)
        n_batches += 1
        
        pbar.set_postfix({
            'loss': f'{total_loss/n_batches:.4f}',
            'acc': f'{total_acc/n_batches:.4f}'
        })
    
    return total_loss / n_batches, total_acc / n_batches

def validate(model, val_dataloader, criterion, device, adaptation_steps, inner_lr):
    model.eval()
    total_val_loss = 0
    total_val_acc = 0
    num_batches = 0

    for episodes in val_dataloader:
        batch_loss = 0
        batch_acc = 0
        
        for episode in episodes:
            support_images = episode['support_images'].to(device, non_blocking=True)
            support_labels = episode['support_labels'].unsqueeze(1).to(device, non_blocking=True)
            query_images = episode['query_images'].to(device, non_blocking=True)
            query_labels = episode['query_labels'].unsqueeze(1).to(device, non_blocking=True)

            adapted_model = copy.deepcopy(model)
            adapted_model.train()
            
            for param in adapted_model.parameters():
                param.requires_grad_(True)

            for _ in range(adaptation_steps):
                support_preds = adapted_model(support_images)
                support_loss = criterion(
                    support_preds[:, 1],
                    support_labels.squeeze(1).float()
                )
                
                grads = torch.autograd.grad(
                    support_loss,
                    adapted_model.parameters(),
                    create_graph=True,
                    retain_graph=True
                )
                
                for param, grad in zip(adapted_model.parameters(), grads):
                    param.data = param.data - inner_lr * grad

            adapted_model.eval()
            with torch.no_grad():
                query_preds = adapted_model(query_images)
                query_loss = criterion(
                    query_preds[:, 1],
                    query_labels.squeeze(1).float()
                )
                query_acc = accuracy(query_preds, query_labels)
                
                batch_loss += query_loss.item()
                batch_acc += query_acc.item()
        
        avg_batch_loss = batch_loss / len(episodes)
        avg_batch_acc = batch_acc / len(episodes)
        
        total_val_loss += avg_batch_loss
        total_val_acc += avg_batch_acc
        num_batches += 1

    return total_val_loss / num_batches, total_val_acc / num_batches

def test_model(model, test_loader, device, test_adaptation_steps, inner_lr):
    try:
        model.eval()
        total_loss = 0
        total_acc = 0
        num_batches = 0
        
        for episodes in tqdm(test_loader, desc="Testing"):
            batch_loss = 0
            batch_acc = 0
            
            for episode in episodes:
                try:
                    adapted_model = copy.deepcopy(model)
                    adapted_model.train()
                    
                    support_images = episode['support_images'].to(device, non_blocking=True)
                    support_labels = episode['support_labels'].unsqueeze(1).to(device, non_blocking=True)
                    query_images = episode['query_images'].to(device, non_blocking=True)
                    query_labels = episode['query_labels'].unsqueeze(1).to(device, non_blocking=True)
                    
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
                        
                        for param, grad in zip(adapted_model.parameters(), grads):
                            if grad is not None:
                                param.data = param.data - inner_lr * grad
                    
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
                        print("WARNING: GPU OOM error. Skipping batch...")
                        torch.cuda.empty_cache()
                        continue
                    raise e
            
            avg_batch_loss = batch_loss / len(episodes)
            avg_batch_acc = batch_acc / len(episodes)
            
            total_loss += avg_batch_loss
            total_acc += avg_batch_acc
            num_batches += 1
        
        return total_loss / num_batches, total_acc / num_batches
    
    except Exception as e:
        print(f"ERROR: Testing failed with error: {str(e)}")
        return None, None

def main(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    arch_dir = os.path.join(args.output_dir, 'conv6', f'seed_{args.seed}')
    os.makedirs(arch_dir, exist_ok=True)
    
    train_tasks = ['regular', 'lines', 'open', 'wider_line', 'scrambled',
                   'random_color', 'arrows', 'irregular', 'filled', 'original']
    
    train_dataset = SameDifferentDataset(args.data_dir, train_tasks, 'train', support_sizes=[args.support_size])
    val_dataset = SameDifferentDataset(args.data_dir, train_tasks, 'val', support_sizes=[args.support_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                            num_workers=4, pin_memory=True, collate_fn=collate_episodes)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                          num_workers=4, pin_memory=True, collate_fn=collate_episodes)
    
    model = SameDifferentCNN().to(device)
    
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
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
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
                maml, val_loader, F.binary_cross_entropy_with_logits,
                device, args.adaptation_steps, args.inner_lr
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
        
        test_loss, test_acc = test_model(
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/meta_h5/pb',
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
    main(args) 