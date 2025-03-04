import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from meta_baseline.models.conv2lr import SameDifferentCNN as MetaCNN
from baselines.models.conv2 import SameDifferentCNN as BaselineCNN
import seaborn as sns

class LinearProbe(nn.Module):
    """
    A simple linear probe for analyzing neural network layer representations.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super(LinearProbe, self).__init__()
        
        if hidden_dim is not None:
            # Non-linear probe with one hidden layer
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.is_linear = False
        else:
            # Linear probe
            self.fc = nn.Linear(input_dim, output_dim)
            self.is_linear = True
    
    def forward(self, x):
        if self.is_linear:
            return self.fc(x)
        else:
            x = F.relu(self.fc1(x))
            return self.fc2(x)

def collect_activations_with_labels(model, data_loader, layer_names, device='cpu'):
    """
    Collect activations and corresponding labels from specified layers of the model.
    
    Args:
        model: The neural network model
        data_loader: DataLoader providing input samples and labels
        layer_names: List of layer names to collect activations from
        device: Torch device
        
    Returns:
        activations_dict: Dictionary mapping layer names to activation matrices
        labels: Tensor of corresponding labels
    """
    activations_dict = {name: [] for name in layer_names}
    all_labels = []
    hooks = []
    
    # Define hook function
    def get_activation(name):
        def hook(module, input, output):
            # Store a copy of the outputs
            activations_dict[name].append(output.detach().cpu())
        return hook
    
    # Register hooks
    for name in layer_names:
        for module_name, module in model.named_modules():
            if name == module_name:
                hooks.append(module.register_forward_hook(get_activation(name)))
    
    # Pass data through the model to collect activations
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Collecting activations"):
            images = batch['image'].to(device)
            labels = batch['label']
            _ = model(images)
            all_labels.append(labels)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Process collected activations
    for name in layer_names:
        if activations_dict[name]:
            # Concatenate all batches
            activations_dict[name] = torch.cat(activations_dict[name], dim=0)
            
            # For 4D activations (conv layers), flatten spatial dimensions
            if len(activations_dict[name].shape) == 4:
                b, c, h, w = activations_dict[name].shape
                activations_dict[name] = activations_dict[name].reshape(b, c, h*w)
                activations_dict[name] = activations_dict[name].reshape(b, -1)
    
    # Concatenate all labels
    all_labels = torch.cat(all_labels, dim=0)
    
    return activations_dict, all_labels

def train_probe(X_train, y_train, X_val, y_val, output_dim, hidden_dim=None, 
                epochs=30, batch_size=128, lr=0.001, device='cpu'):
    """
    Train a linear probe on collected activations.
    
    Args:
        X_train: Activation features for training
        y_train: Labels for training
        X_val: Activation features for validation
        y_val: Labels for validation
        output_dim: Number of output classes
        hidden_dim: Size of hidden layer, None for linear probe
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        device: Torch device
        
    Returns:
        probe: Trained probe model
        history: Training history (loss and accuracy)
    """
    # Move data to device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    
    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    
    # Initialize model
    input_dim = X_train.shape[1]
    probe = LinearProbe(input_dim, output_dim, hidden_dim).to(device)
    
    # Initialize optimizer and criterion
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Train
    for epoch in range(epochs):
        probe.train()
        train_loss = 0.0
        train_correct = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            outputs = probe(X_batch)
            loss = criterion(outputs, y_batch.long())
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == y_batch).sum().item()
        
        # Calculate training metrics
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / len(train_dataset)
        
        # Evaluate on validation set
        probe.eval()
        with torch.no_grad():
            val_outputs = probe(X_val)
            val_loss = criterion(val_outputs, y_val.long()).item()
            _, val_predicted = torch.max(val_outputs, 1)
            val_acc = (val_predicted == y_val).float().mean().item()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    return probe, history

def probe_layer_representations(model, data_loader, layer_names, labels, output_dir, device='cpu'):
    """
    Train probes for different layers to analyze representation quality.
    
    Args:
        model: The neural network model
        data_loader: DataLoader providing input samples
        layer_names: List of layer names to analyze
        labels: Labels for probing tasks (dictionary mapping task_name to label tensor)
        output_dir: Directory to save results
        device: Torch device
        
    Returns:
        results: Dictionary of probing results
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    # Collect activations
    activations_dict, _ = collect_activations_with_labels(model, data_loader, layer_names, device)
    
    # For each layer and each task
    for layer_name in layer_names:
        layer_results = {}
        activations = activations_dict[layer_name]
        
        for task_name, task_labels in labels.items():
            print(f"Probing {layer_name} for task: {task_name}")
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                activations, task_labels, test_size=0.2, random_state=42
            )
            
            # Get output dimension
            if task_labels.dim() > 1:
                output_dim = task_labels.shape[1]
            else:
                output_dim = len(torch.unique(task_labels))
            
            # Train linear probe
            linear_probe, linear_history = train_probe(
                X_train, y_train, X_val, y_val, 
                output_dim=output_dim, 
                hidden_dim=None,
                device=device
            )
            
            # Train non-linear probe (MLP)
            nonlinear_probe, nonlinear_history = train_probe(
                X_train, y_train, X_val, y_val, 
                output_dim=output_dim, 
                hidden_dim=min(activations.shape[1], 128),
                device=device
            )
            
            # Plot training history
            plt.figure(figsize=(12, 5))
            
            # Plot loss
            plt.subplot(1, 2, 1)
            plt.plot(linear_history['train_loss'], label='Linear - Train')
            plt.plot(linear_history['val_loss'], label='Linear - Val')
            plt.plot(nonlinear_history['train_loss'], label='MLP - Train')
            plt.plot(nonlinear_history['val_loss'], label='MLP - Val')
            plt.title(f'Loss - {layer_name} - {task_name}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Plot accuracy
            plt.subplot(1, 2, 2)
            plt.plot(linear_history['train_acc'], label='Linear - Train')
            plt.plot(linear_history['val_acc'], label='Linear - Val')
            plt.plot(nonlinear_history['train_acc'], label='MLP - Train')
            plt.plot(nonlinear_history['val_acc'], label='MLP - Val')
            plt.title(f'Accuracy - {layer_name} - {task_name}')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'probe_{layer_name.replace(".", "_")}_{task_name}.png'))
            plt.close()
            
            # Store results
            layer_results[task_name] = {
                'linear': {
                    'probe': linear_probe,
                    'history': linear_history,
                    'final_val_acc': linear_history['val_acc'][-1]
                },
                'nonlinear': {
                    'probe': nonlinear_probe,
                    'history': nonlinear_history,
                    'final_val_acc': nonlinear_history['val_acc'][-1]
                }
            }
        
        results[layer_name] = layer_results
    
    # Create summary plot of probe accuracy across layers
    summarize_probe_results(results, output_dir)
    
    return results

def summarize_probe_results(results, output_dir):
    """
    Create summary visualizations of probing results across layers.
    
    Args:
        results: Dictionary of probing results
        output_dir: Directory to save visualizations
    """
    layer_names = list(results.keys())
    task_names = list(results[layer_names[0]].keys())
    
    # Create dataframes for linear and non-linear accuracies
    linear_accs = np.zeros((len(layer_names), len(task_names)))
    nonlinear_accs = np.zeros((len(layer_names), len(task_names)))
    
    for i, layer in enumerate(layer_names):
        for j, task in enumerate(task_names):
            linear_accs[i, j] = results[layer][task]['linear']['final_val_acc']
            nonlinear_accs[i, j] = results[layer][task]['nonlinear']['final_val_acc']
    
    # Plot heatmaps
    plt.figure(figsize=(18, 8))
    
    # Linear probes
    plt.subplot(1, 2, 1)
    sns.heatmap(linear_accs, annot=True, fmt=".3f", cmap="YlGnBu",
                xticklabels=task_names, yticklabels=layer_names)
    plt.title('Linear Probe Accuracy by Layer and Task')
    plt.xlabel('Task')
    plt.ylabel('Layer')
    
    # Non-linear probes
    plt.subplot(1, 2, 2)
    sns.heatmap(nonlinear_accs, annot=True, fmt=".3f", cmap="YlGnBu",
                xticklabels=task_names, yticklabels=layer_names)
    plt.title('Non-linear Probe Accuracy by Layer and Task')
    plt.xlabel('Task')
    plt.ylabel('Layer')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'probe_accuracy_summary.png'))
    plt.close()
    
    # Plot accuracy by layer depth
    plt.figure(figsize=(15, 6))
    
    for j, task in enumerate(task_names):
        linear_task_accs = [results[layer][task]['linear']['final_val_acc'] for layer in layer_names]
        nonlinear_task_accs = [results[layer][task]['nonlinear']['final_val_acc'] for layer in layer_names]
        
        plt.plot(range(len(layer_names)), linear_task_accs, 'o-', label=f'{task} - Linear')
        plt.plot(range(len(layer_names)), nonlinear_task_accs, 's--', label=f'{task} - MLP')
    
    plt.xticks(range(len(layer_names)), layer_names, rotation=45)
    plt.title('Probe Accuracy by Layer Depth')
    plt.xlabel('Layer')
    plt.ylabel('Validation Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'probe_accuracy_by_depth.png'))
    plt.close()

def compare_meta_and_baseline_representations(meta_model, baseline_model, data_loader, 
                                             probe_tasks, output_dir, device='cpu'):
    """
    Compare the quality of representations between meta-learned and baseline models.
    
    Args:
        meta_model: Meta-learned model
        baseline_model: Baseline model
        data_loader: DataLoader providing input samples and labels
        probe_tasks: Dictionary mapping task names to labels
        output_dir: Directory to save results
        device: Torch device
        
    Returns:
        combined_results: Dictionary of probing results for both models
    """
    meta_output_dir = os.path.join(output_dir, 'meta_probing')
    baseline_output_dir = os.path.join(output_dir, 'baseline_probing')
    os.makedirs(meta_output_dir, exist_ok=True)
    os.makedirs(baseline_output_dir, exist_ok=True)
    
    # Define layers to analyze
    meta_layers = ['conv1', 'conv2', 'fc_layers.0', 'fc_layers.1', 'fc_layers.2']
    baseline_layers = ['conv1', 'conv2', 'fc1', 'fc2', 'fc3']
    
    # Run probing for meta model
    print("Probing meta-learned model representations...")
    meta_results = probe_layer_representations(
        meta_model, data_loader, meta_layers, probe_tasks, meta_output_dir, device
    )
    
    # Run probing for baseline model
    print("Probing baseline model representations...")
    baseline_results = probe_layer_representations(
        baseline_model, data_loader, baseline_layers, probe_tasks, baseline_output_dir, device
    )
    
    # Create comparison visualizations
    compare_probing_results(meta_results, baseline_results, 
                           meta_layers, baseline_layers,
                           output_dir)
    
    return {
        'meta': meta_results,
        'baseline': baseline_results
    }

def compare_probing_results(meta_results, baseline_results, 
                          meta_layers, baseline_layers,
                          output_dir):
    """
    Create visualizations comparing probing results between models.
    
    Args:
        meta_results: Probing results for meta-learned model
        baseline_results: Probing results for baseline model
        meta_layers: Layer names for meta-learned model
        baseline_layers: Layer names for baseline model
        output_dir: Directory to save visualizations
    """
    # Ensure we have tasks that are common to both model results
    task_names = list(meta_results[meta_layers[0]].keys())
    
    # Compare linear probe performance
    plt.figure(figsize=(15, 6))
    
    for j, task in enumerate(task_names):
        meta_accs = [meta_results[layer][task]['linear']['final_val_acc'] for layer in meta_layers]
        baseline_accs = [baseline_results[layer][task]['linear']['final_val_acc'] for layer in baseline_layers]
        
        plt.plot(range(len(meta_layers)), meta_accs, 'o-', 
                 label=f'{task} - Meta-learned')
        plt.plot(range(len(baseline_layers)), baseline_accs, 's--', 
                 label=f'{task} - Baseline')
    
    # Use corresponding layers for x-axis labels (since they might differ in name)
    x_labels = []
    for i in range(len(meta_layers)):
        if i < len(baseline_layers):
            meta_name = meta_layers[i].split('.')[-1] if '.' in meta_layers[i] else meta_layers[i]
            baseline_name = baseline_layers[i].split('.')[-1] if '.' in baseline_layers[i] else baseline_layers[i]
            if meta_name == baseline_name:
                x_labels.append(meta_name)
            else:
                x_labels.append(f"{meta_name}/{baseline_name}")
        else:
            x_labels.append(meta_layers[i])
    
    plt.xticks(range(len(x_labels)), x_labels, rotation=45)
    plt.title('Linear Probe Accuracy Comparison: Meta-learned vs Baseline')
    plt.xlabel('Layer')
    plt.ylabel('Validation Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'meta_vs_baseline_linear_probe.png'))
    plt.close()
    
    # Compare non-linear probe performance
    plt.figure(figsize=(15, 6))
    
    for j, task in enumerate(task_names):
        meta_accs = [meta_results[layer][task]['nonlinear']['final_val_acc'] for layer in meta_layers]
        baseline_accs = [baseline_results[layer][task]['nonlinear']['final_val_acc'] for layer in baseline_layers]
        
        plt.plot(range(len(meta_layers)), meta_accs, 'o-', 
                 label=f'{task} - Meta-learned')
        plt.plot(range(len(baseline_layers)), baseline_accs, 's--', 
                 label=f'{task} - Baseline')
    
    plt.xticks(range(len(x_labels)), x_labels, rotation=45)
    plt.title('Non-linear Probe Accuracy Comparison: Meta-learned vs Baseline')
    plt.xlabel('Layer')
    plt.ylabel('Validation Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'meta_vs_baseline_nonlinear_probe.png'))
    plt.close()
    
    # Compare meta vs baseline performance difference
    plt.figure(figsize=(15, 6))
    
    for j, task in enumerate(task_names):
        meta_accs = [meta_results[layer][task]['linear']['final_val_acc'] for layer in meta_layers]
        baseline_accs = [baseline_results[layer][task]['linear']['final_val_acc'] for layer in baseline_layers]
        
        # Calculate difference (meta - baseline)
        diff_accs = [m - b for m, b in zip(meta_accs, baseline_accs)]
        
        plt.plot(range(len(diff_accs)), diff_accs, 'o-', 
                 label=f'{task}')
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xticks(range(len(x_labels)), x_labels, rotation=45)
    plt.title('Meta-learning Advantage (Meta - Baseline) by Layer')
    plt.xlabel('Layer')
    plt.ylabel('Accuracy Difference')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'meta_advantage_by_layer.png'))
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Probe neural network representations')
    parser.add_argument('--meta-model', type=str, required=True, 
                        help='Path to meta-learned model checkpoint')
    parser.add_argument('--baseline-model', type=str, required=True,
                        help='Path to baseline model checkpoint') 
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing dataset for activation collection')
    parser.add_argument('--output-dir', type=str, default='probing_results',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # This part would be implemented to load your dataset and models
    # For a complete implementation, you would need to adapt this to your specific data loading code
    
    print("Probing analysis completed.") 