import os
import torch
import torch.nn.functional as F
import json
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import copy
import gc
from pathlib import Path
import sys
import learn2learn as l2l
import torch.nn as nn

from models.conv2lr import SameDifferentCNN as Conv2CNN
from models.conv4lr import SameDifferentCNN as Conv4CNN
from models.conv6lr import SameDifferentCNN as Conv6CNN
from models.conv6lr import SameDifferentDataset, collate_episodes

PB_TASKS = [
    'regular', 'lines', 'open', 'wider_line', 'scrambled',
    'random_color', 'arrows', 'irregular', 'filled', 'original'
]
ARCHITECTURES = {
    'conv2': Conv2CNN,
    'conv4': Conv4CNN,
    'conv6': Conv6CNN
}

def accuracy(predictions, targets):
    """Calculate binary classification accuracy using second logit."""
    predicted_labels = (predictions[:, 1] > 0.0).float()
    return (predicted_labels == targets.squeeze(1)).float().mean()

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train_epoch(model, train_loader, optimizer, device, adaptation_steps, scaler):
    """Single training epoch with mixed precision and gradient monitoring."""
    model.train()
    total_loss = 0
    total_acc = 0
    n_batches = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, episodes in enumerate(pbar):
        optimizer.zero_grad()
        batch_loss = 0
        batch_acc = 0
        
        with torch.amp.autocast(device_type='cuda'):
            for episode in episodes:
                learner = model.clone()
                
                support_images = episode['support_images'].to(device, non_blocking=True)
                support_labels = episode['support_labels'].unsqueeze(1).to(device, non_blocking=True)
                query_images = episode['query_images'].to(device, non_blocking=True)
                query_labels = episode['query_labels'].unsqueeze(1).to(device, non_blocking=True)
                
                for step in range(adaptation_steps):
                    support_preds = learner(support_images)
                    support_loss = F.binary_cross_entropy_with_logits(
                        support_preds[:, 1], support_labels.squeeze(1).float())
                    
                    if support_loss.item() > 10 and step == 0:
                        print(f"\nWARNING: High support loss: {support_loss.item():.4f}")
                        print("Support predictions:", torch.sigmoid(support_preds[:, 1]).detach().cpu().numpy())
                        print("Support labels:", support_labels.squeeze(1).cpu().numpy())
                    
                    trainable_params = [p for p in learner.parameters() if p.requires_grad]
                    grads = torch.autograd.grad(
                        support_loss,
                        trainable_params,
                        create_graph=True,
                        allow_unused=True,
                        retain_graph=True
                    )
                    
                    param_grad_pairs = [(p, g) for p, g in zip(trainable_params, grads) if g is not None]
                    if not param_grad_pairs:
                        print("\nWARNING: No valid gradients found in inner loop")
                        continue
                        
                    params, filtered_grads = zip(*param_grad_pairs)
                    
                    grad_norm = torch.norm(torch.stack([torch.norm(g) for g in filtered_grads]))
                    if grad_norm > 10:
                        scaling_factor = 10 / grad_norm
                        filtered_grads = [g * scaling_factor for g in filtered_grads]
                    
                    for param, grad in zip(params, filtered_grads):
                        param.data = param.data - learner.lr * grad
                
                query_preds = learner(query_images)
                query_loss = F.binary_cross_entropy_with_logits(
                    query_preds[:, 1], query_labels.squeeze(1).float())
                query_acc = accuracy(query_preds, query_labels)
                
                batch_loss += query_loss
                batch_acc += query_acc
        
        batch_loss = batch_loss / len(episodes)
        batch_acc = batch_acc / len(episodes)
        
        scaler.scale(batch_loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=10.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += batch_loss.item()
        total_acc += batch_acc.item()
        n_batches += 1
        
        pbar.set_postfix({
            'loss': total_loss/n_batches,
            'acc': total_acc/n_batches
        })
    
    return total_loss/n_batches, total_acc/n_batches

def validate(model, val_loader, device, adaptation_steps, inner_lr):
    """Validate using MAML with gradient monitoring."""
    model.eval()
    total_val_loss = 0
    total_val_acc = 0
    num_batches = 0

    for batch_idx, episodes in enumerate(val_loader):
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

            for step in range(adaptation_steps):
                support_preds = adapted_model(support_images)
                support_loss = F.binary_cross_entropy_with_logits(
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
                query_loss = F.binary_cross_entropy_with_logits(
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

    avg_val_loss = total_val_loss / num_batches
    avg_val_acc = total_val_acc / num_batches

    return avg_val_loss, avg_val_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/pb/pb',
                        help='Directory containing the PB dataset')
    parser.add_argument('--output_dir', type=str, default='results/meta_baselines',
                        help='Directory to save results')
    parser.add_argument('--architecture', type=str, required=True,
                        choices=['conv2', 'conv4', 'conv6'],
                        help='Architecture to train')
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
        
        set_seed(args.seed)
        
        arch_dir = os.path.join(args.output_dir, args.architecture, f'seed_{args.seed}')
        os.makedirs(arch_dir, exist_ok=True)
        
        print("\nCreating datasets...")
        train_dataset = SameDifferentDataset(args.data_dir, PB_TASKS, 'train', support_sizes=[args.support_size])
        val_dataset = SameDifferentDataset(args.data_dir, PB_TASKS, 'val', support_sizes=[args.support_size])
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                num_workers=4, pin_memory=True, collate_fn=collate_episodes)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                              num_workers=4, pin_memory=True, collate_fn=collate_episodes)
        
        print(f"\nCreating {args.architecture} model")
        model = ARCHITECTURES[args.architecture]().to(device)
        
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
        for task in PB_TASKS:
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