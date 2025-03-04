import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from meta_baseline.models.conv2lr import SameDifferentCNN as MetaCNN
from baselines.models.conv2 import SameDifferentCNN as BaselineCNN
import torch.nn as nn

def load_models(meta_model_path, baseline_model_path, device=torch.device('cpu')):
    """
    Load both meta-learned and baseline models for comparison.
    
    Args:
        meta_model_path: Path to the saved meta-learned model checkpoint
        baseline_model_path: Path to the saved baseline model checkpoint
        device: Torch device to load models to
        
    Returns:
        meta_model: Loaded meta-learned model
        baseline_model: Loaded baseline model
    """
    meta_model = MetaCNN().to(device)
    baseline_model = BaselineCNN().to(device)
    
    # Load meta model
    meta_checkpoint = torch.load(meta_model_path, map_location=device)
    # Check if it's a MAML checkpoint with model_state_dict
    if 'model_state_dict' in meta_checkpoint:
        meta_model.load_state_dict(meta_checkpoint['model_state_dict'])
    else:
        meta_model.load_state_dict(meta_checkpoint)
    
    # Load baseline model
    baseline_checkpoint = torch.load(baseline_model_path, map_location=device)
    if 'model_state_dict' in baseline_checkpoint:
        baseline_model.load_state_dict(baseline_checkpoint['model_state_dict'])
    else:
        baseline_model.load_state_dict(baseline_checkpoint)
    
    return meta_model, baseline_model

def extract_weights(model, layer_type=nn.Conv2d):
    """
    Extract weights from specific layer types in a model.
    
    Args:
        model: The PyTorch model to extract weights from
        layer_type: Type of layer to extract weights from
        
    Returns:
        weights_dict: Dictionary mapping layer names to flattened weight tensors
    """
    weights_dict = {}
    
    for name, module in model.named_modules():
        if isinstance(module, layer_type):
            # Get the weight tensor and flatten it
            weight = module.weight.data.cpu().numpy()
            weights_dict[name] = weight.reshape(weight.shape[0], -1)
    
    return weights_dict

def run_pca_on_weights(weights_dict, n_components=2):
    """
    Perform PCA on layer weights.
    
    Args:
        weights_dict: Dictionary mapping layer names to weight matrices
        n_components: Number of PCA components
        
    Returns:
        pca_results: Dictionary with PCA results for each layer
    """
    pca_results = {}
    
    for layer_name, weights in weights_dict.items():
        # Standardize the weights
        weights_centered = weights - np.mean(weights, axis=0)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(weights_centered)
        
        pca_results[layer_name] = {
            'transformed': pca_result,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'components': pca.components_
        }
    
    return pca_results

def compare_weight_spaces(meta_model, baseline_model, output_dir='weight_analysis_results'):
    """
    Compare the weight spaces of meta-learned vs baseline models.
    
    Args:
        meta_model: The meta-learned model
        baseline_model: The baseline model
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract weights from convolutional layers
    meta_conv_weights = extract_weights(meta_model, nn.Conv2d)
    baseline_conv_weights = extract_weights(baseline_model, nn.Conv2d)
    
    # Extract weights from fully connected layers
    meta_fc_weights = extract_weights(meta_model, nn.Linear)
    baseline_fc_weights = extract_weights(baseline_model, nn.Linear)
    
    # Run PCA on convolutional weights
    meta_conv_pca = run_pca_on_weights(meta_conv_weights)
    baseline_conv_pca = run_pca_on_weights(baseline_conv_weights)
    
    # Run PCA on fully connected weights
    meta_fc_pca = run_pca_on_weights(meta_fc_weights)
    baseline_fc_pca = run_pca_on_weights(baseline_fc_weights)
    
    # Plot and compare PCA results
    plot_pca_comparisons(meta_conv_pca, baseline_conv_pca, 'conv', output_dir)
    plot_pca_comparisons(meta_fc_pca, baseline_fc_pca, 'fc', output_dir)
    
    # Calculate and plot weight statistics
    plot_weight_statistics(meta_model, baseline_model, output_dir)
    
    return {
        'meta_conv_pca': meta_conv_pca,
        'baseline_conv_pca': baseline_conv_pca,
        'meta_fc_pca': meta_fc_pca,
        'baseline_fc_pca': baseline_fc_pca
    }

def plot_pca_comparisons(meta_pca, baseline_pca, layer_type, output_dir):
    """
    Create plots comparing PCA projections of weights between models.
    
    Args:
        meta_pca: PCA results for meta-learned model
        baseline_pca: PCA results for baseline model
        layer_type: String identifier for the layer type
        output_dir: Directory to save plots
    """
    # Find common layers between models
    common_layers = set(meta_pca.keys()).intersection(set(baseline_pca.keys()))
    
    for layer in common_layers:
        plt.figure(figsize=(15, 6))
        
        # Plot meta-learned model weights
        plt.subplot(1, 2, 1)
        plt.scatter(
            meta_pca[layer]['transformed'][:, 0],
            meta_pca[layer]['transformed'][:, 1],
            alpha=0.7
        )
        plt.title(f'Meta-Learned Model: {layer}')
        plt.xlabel(f'PC1 ({meta_pca[layer]["explained_variance_ratio"][0]:.2%} var)')
        plt.ylabel(f'PC2 ({meta_pca[layer]["explained_variance_ratio"][1]:.2%} var)')
        plt.grid(alpha=0.3)
        
        # Plot baseline model weights
        plt.subplot(1, 2, 2)
        plt.scatter(
            baseline_pca[layer]['transformed'][:, 0],
            baseline_pca[layer]['transformed'][:, 1],
            alpha=0.7,
            color='orangered'
        )
        plt.title(f'Baseline Model: {layer}')
        plt.xlabel(f'PC1 ({baseline_pca[layer]["explained_variance_ratio"][0]:.2%} var)')
        plt.ylabel(f'PC2 ({baseline_pca[layer]["explained_variance_ratio"][1]:.2%} var)')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'pca_comparison_{layer_type}_{layer.replace(".", "_")}.png'))
        plt.close()

def plot_weight_statistics(meta_model, baseline_model, output_dir):
    """
    Plot statistics of weights (distribution, sparsity) between models.
    
    Args:
        meta_model: The meta-learned model
        baseline_model: The baseline model
        output_dir: Directory to save plots
    """
    # Extract all weights from both models
    meta_weights = {}
    baseline_weights = {}
    
    for name, param in meta_model.named_parameters():
        if 'weight' in name:
            meta_weights[name] = param.data.cpu().numpy().flatten()
    
    for name, param in baseline_model.named_parameters():
        if 'weight' in name:
            baseline_weights[name] = param.data.cpu().numpy().flatten()
    
    # Find common layers
    common_layers = set(meta_weights.keys()).intersection(set(baseline_weights.keys()))
    
    for layer in common_layers:
        plt.figure(figsize=(15, 6))
        
        # Plot weight distributions
        plt.subplot(1, 2, 1)
        sns.kdeplot(meta_weights[layer], label='Meta-Learned', fill=True, alpha=0.3)
        sns.kdeplot(baseline_weights[layer], label='Baseline', fill=True, alpha=0.3)
        plt.title(f'Weight Distribution: {layer}')
        plt.xlabel('Weight Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Plot weight sparsity (histogram of absolute values)
        plt.subplot(1, 2, 2)
        sns.histplot(np.abs(meta_weights[layer]), 
                   label='Meta-Learned', 
                   alpha=0.5, 
                   bins=50,
                   kde=True,
                   stat='density')
        sns.histplot(np.abs(baseline_weights[layer]), 
                   label='Baseline', 
                   alpha=0.5, 
                   bins=50,
                   kde=True,
                   stat='density')
        plt.title(f'Weight Magnitude Distribution: {layer}')
        plt.xlabel('Absolute Weight Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'weight_stats_{layer.replace(".", "_")}.png'))
        plt.close()

def compute_weight_correlations(meta_model, baseline_model, output_dir):
    """
    Compute and visualize weight correlations across all layers between models.
    
    Args:
        meta_model: The meta-learned model
        baseline_model: The baseline model
        output_dir: Directory to save results
    """
    # Process model and create correlation matrix of weight vectors
    # (Implementation will depend on how you want to construct the correlation matrix)
    pass

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze weight spaces of models')
    parser.add_argument('--meta-model', type=str, required=True, 
                        help='Path to meta-learned model checkpoint')
    parser.add_argument('--baseline-model', type=str, required=True,
                        help='Path to baseline model checkpoint')
    parser.add_argument('--output-dir', type=str, default='weight_analysis_results',
                        help='Directory to save analysis results')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    
    # Load models
    meta_model, baseline_model = load_models(
        args.meta_model, 
        args.baseline_model,
        device
    )
    
    # Run analysis
    compare_weight_spaces(meta_model, baseline_model, args.output_dir)
    
    print(f"Weight space analysis completed. Results saved to {args.output_dir}") 