import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from meta_baseline.models.conv2lr import SameDifferentCNN as MetaCNN
from baselines.models.conv2 import SameDifferentCNN as BaselineCNN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

class SparseAutoEncoder(nn.Module):
    """
    Sparse autoencoder for learning interpretable features from model activations.
    
    This implements a tied-weight autoencoder with L1 regularization for sparsity.
    """
    def __init__(self, input_dim, hidden_dim, l1_coef=1e-4):
        super(SparseAutoEncoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)
        self.l1_coef = l1_coef
        
        # Initialize weights
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        
    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        activations = F.relu(encoded)
        
        # Decode
        decoded = self.decoder(activations)
        
        # Return both decoded output and activations
        return decoded, activations
    
    def get_l1_loss(self, activations):
        """Calculate L1 regularization loss on activations."""
        return self.l1_coef * torch.mean(torch.abs(activations))

def collect_activations(model, data_loader, layer_names, device='cpu'):
    """
    Collect activations from specified layers of the model.
    
    Args:
        model: The neural network model
        data_loader: DataLoader providing input samples
        layer_names: List of layer names to collect activations from
        device: Torch device
        
    Returns:
        activations_dict: Dictionary mapping layer names to activation matrices
    """
    activations_dict = {name: [] for name in layer_names}
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
            _ = model(images)
    
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
    
    return activations_dict

def train_sparse_autoencoder(activations, hidden_dim, l1_coef=1e-4, 
                             epochs=100, batch_size=256, lr=1e-3, device='cpu'):
    """
    Train a sparse autoencoder on collected activations.
    
    Args:
        activations: Tensor of activations [n_samples, n_features]
        hidden_dim: Number of hidden units in the autoencoder
        l1_coef: L1 regularization coefficient for sparsity
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        device: Torch device
        
    Returns:
        model: Trained sparse autoencoder
        losses: Dictionary of training losses
    """
    input_dim = activations.shape[1]
    
    # Create the autoencoder
    model = SparseAutoEncoder(input_dim, hidden_dim, l1_coef).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Create data loader
    dataset = torch.utils.data.TensorDataset(activations)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Track losses
    losses = {'reconstruction': [], 'l1': [], 'total': []}
    
    # Train
    model.train()
    for epoch in tqdm(range(epochs), desc="Training autoencoder"):
        epoch_recon_loss = 0.0
        epoch_l1_loss = 0.0
        epoch_total_loss = 0.0
        
        for batch in loader:
            x = batch[0].to(device)
            
            # Forward pass
            decoded, activations = model(x)
            
            # Calculate losses
            recon_loss = F.mse_loss(decoded, x)
            l1_loss = model.get_l1_loss(activations)
            total_loss = recon_loss + l1_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Track losses
            epoch_recon_loss += recon_loss.item()
            epoch_l1_loss += l1_loss.item()
            epoch_total_loss += total_loss.item()
        
        # Record average losses
        losses['reconstruction'].append(epoch_recon_loss / len(loader))
        losses['l1'].append(epoch_l1_loss / len(loader))
        losses['total'].append(epoch_total_loss / len(loader))
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Loss = {losses['total'][-1]:.6f}, "
                  f"Recon = {losses['reconstruction'][-1]:.6f}, "
                  f"L1 = {losses['l1'][-1]:.6f}")
    
    return model, losses

def analyze_activations(activation_dict, output_dir='activation_analysis'):
    """
    Analyze activations using dimensionality reduction.
    
    Args:
        activation_dict: Dictionary of activations for each layer
        output_dir: Directory to save analysis results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for layer_name, activations in activation_dict.items():
        # Get sample of activations (use all if fewer than 10000)
        if len(activations) > 10000:
            indices = np.random.choice(len(activations), 10000, replace=False)
            act_sample = activations[indices].numpy()
        else:
            act_sample = activations.numpy()
        
        # Apply PCA
        pca = PCA(n_components=2)
        act_pca = pca.fit_transform(act_sample)
        
        # Plot PCA results
        plt.figure(figsize=(10, 8))
        plt.scatter(act_pca[:, 0], act_pca[:, 1], alpha=0.5, s=5)
        plt.title(f'PCA of Activations: {layer_name}')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'pca_activations_{layer_name.replace(".", "_")}.png'))
        plt.close()
        
        # Calculate activation statistics
        act_mean = np.mean(act_sample, axis=0)
        act_std = np.std(act_sample, axis=0)
        act_sparsity = np.mean(act_sample == 0)
        
        # Plot activation statistics
        plt.figure(figsize=(15, 5))
        
        # Mean activation per neuron
        plt.subplot(1, 2, 1)
        plt.plot(act_mean, 'b-', alpha=0.7)
        plt.title(f'Mean Activation per Neuron: {layer_name}')
        plt.xlabel('Neuron Index')
        plt.ylabel('Mean Activation')
        plt.grid(alpha=0.3)
        
        # Standard deviation per neuron
        plt.subplot(1, 2, 2)
        plt.plot(act_std, 'r-', alpha=0.7)
        plt.title(f'Activation Std Dev per Neuron: {layer_name}')
        plt.xlabel('Neuron Index')
        plt.ylabel('StdDev of Activation')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'act_stats_{layer_name.replace(".", "_")}.png'))
        plt.close()
        
        # Save activation statistics
        with open(os.path.join(output_dir, f'act_summary_{layer_name.replace(".", "_")}.txt'), 'w') as f:
            f.write(f"Layer: {layer_name}\n")
            f.write(f"Mean activation: {np.mean(act_sample):.6f}\n")
            f.write(f"StdDev of activation: {np.std(act_sample):.6f}\n")
            f.write(f"Sparsity (fraction of zeros): {act_sparsity:.6f}\n")
            f.write(f"Min activation: {np.min(act_sample):.6f}\n")
            f.write(f"Max activation: {np.max(act_sample):.6f}\n")

def visualize_autoencoder_features(autoencoder, layer_name, output_dir):
    """
    Visualize the features learned by the sparse autoencoder.
    
    Args:
        autoencoder: Trained sparse autoencoder
        layer_name: Name of the layer for the filename
        output_dir: Output directory for visualizations
    """
    # Get the encoder weights
    encoder_weights = autoencoder.encoder.weight.data.cpu().numpy()
    
    # Normalize weights for better visualization
    normalized_weights = (encoder_weights - encoder_weights.min()) / (encoder_weights.max() - encoder_weights.min())
    
    # Plot distribution of weights
    plt.figure(figsize=(10, 6))
    plt.hist(encoder_weights.flatten(), bins=50, alpha=0.7)
    plt.title(f'Distribution of Encoder Weights: {layer_name}')
    plt.xlabel('Weight Value')
    plt.ylabel('Count')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'ae_weight_dist_{layer_name.replace(".", "_")}.png'))
    plt.close()
    
    # Calculate feature sparsity
    feature_sparsity = np.mean(np.abs(encoder_weights) < 1e-3, axis=1)
    
    # Plot feature sparsity
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_sparsity)), feature_sparsity, alpha=0.7)
    plt.title(f'Feature Sparsity: {layer_name}')
    plt.xlabel('Feature Index')
    plt.ylabel('Sparsity')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'ae_feature_sparsity_{layer_name.replace(".", "_")}.png'))
    plt.close()
    
    return normalized_weights

def compare_models_with_autoencoders(meta_model, baseline_model, dataloader, device='cpu'):
    """
    Compare meta-learned and baseline models using sparse autoencoders.
    
    Args:
        meta_model: Meta-learned model
        baseline_model: Baseline model
        dataloader: DataLoader providing samples
        device: Torch device
        
    Returns:
        Dictionary of results and comparisons
    """
    # Define layers to analyze
    meta_layers = ['conv1', 'conv2', 'fc_layers.0', 'fc_layers.1', 'fc_layers.2']
    baseline_layers = ['conv1', 'conv2', 'fc1', 'fc2', 'fc3']
    
    # Collect activations
    print("Collecting meta-model activations...")
    meta_activations = collect_activations(meta_model, dataloader, meta_layers, device)
    
    print("Collecting baseline model activations...")
    baseline_activations = collect_activations(baseline_model, dataloader, baseline_layers, device)
    
    results = {
        'meta_autoencoders': {},
        'baseline_autoencoders': {}
    }
    
    output_dir = 'autoencoder_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Train autoencoders for meta model
    for layer_name, activations in meta_activations.items():
        print(f"Training autoencoder for meta model layer: {layer_name}")
        hidden_dim = min(activations.shape[1] // 2, 256)  # Smaller dimension for compression
        
        # Move activations to device
        activations = activations.to(device)
        
        # Train autoencoder
        autoencoder, losses = train_sparse_autoencoder(
            activations, hidden_dim, l1_coef=1e-4, epochs=100, device=device
        )
        
        # Save autoencoder
        torch.save(autoencoder.state_dict(), 
                  os.path.join(output_dir, f'meta_ae_{layer_name.replace(".", "_")}.pt'))
        
        # Plot training losses
        plt.figure(figsize=(10, 6))
        plt.plot(losses['total'], label='Total Loss')
        plt.plot(losses['reconstruction'], label='Reconstruction Loss')
        plt.plot(losses['l1'], label='L1 Regularization')
        plt.title(f'Autoencoder Training Losses: Meta Model - {layer_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'meta_ae_losses_{layer_name.replace(".", "_")}.png'))
        plt.close()
        
        # Visualize learned features
        feature_weights = visualize_autoencoder_features(autoencoder, f'meta_{layer_name}', output_dir)
        
        # Store results
        results['meta_autoencoders'][layer_name] = {
            'autoencoder': autoencoder,
            'losses': losses,
            'feature_weights': feature_weights
        }
    
    # Train autoencoders for baseline model
    for layer_name, activations in baseline_activations.items():
        print(f"Training autoencoder for baseline model layer: {layer_name}")
        hidden_dim = min(activations.shape[1] // 2, 256)  # Smaller dimension for compression
        
        # Move activations to device
        activations = activations.to(device)
        
        # Train autoencoder
        autoencoder, losses = train_sparse_autoencoder(
            activations, hidden_dim, l1_coef=1e-4, epochs=100, device=device
        )
        
        # Save autoencoder
        torch.save(autoencoder.state_dict(), 
                  os.path.join(output_dir, f'baseline_ae_{layer_name.replace(".", "_")}.pt'))
        
        # Plot training losses
        plt.figure(figsize=(10, 6))
        plt.plot(losses['total'], label='Total Loss')
        plt.plot(losses['reconstruction'], label='Reconstruction Loss')
        plt.plot(losses['l1'], label='L1 Regularization')
        plt.title(f'Autoencoder Training Losses: Baseline Model - {layer_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'baseline_ae_losses_{layer_name.replace(".", "_")}.png'))
        plt.close()
        
        # Visualize learned features
        feature_weights = visualize_autoencoder_features(autoencoder, f'baseline_{layer_name}', output_dir)
        
        # Store results
        results['baseline_autoencoders'][layer_name] = {
            'autoencoder': autoencoder,
            'losses': losses,
            'feature_weights': feature_weights
        }
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze models using sparse autoencoders')
    parser.add_argument('--meta-model', type=str, required=True, 
                        help='Path to meta-learned model checkpoint')
    parser.add_argument('--baseline-model', type=str, required=True,
                        help='Path to baseline model checkpoint')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing dataset for activation collection')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # This part would be implemented to load your dataset and models
    # For a complete implementation, you would need to adapt this to your specific data loading code
    
    print("Sparse autoencoder analysis completed.") 