import sys
sys.path.append('../src')

import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.templates.default = 'plotly_white'
from sklearn.decomposition import PCA
import os
from pathlib import Path

def load_model_weights(filepath):
    """Load weights from a model file."""
    state_dict = torch.load(filepath, map_location='cpu')
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        return state_dict['model_state_dict']
    return state_dict

# Weights directory
weights_dir = '../weights'
model_files = {
    'conv2': ('best_model_vanilla2.pt', 'best_model_meta2.pt'),
    'conv4': ('best_model_vanilla4.pt', 'best_model_meta4.pt'),
    'conv6': ('best_model_vanilla6.pt', 'best_model_meta6.pt')
}

# Create output directory
os.makedirs('pca_plots/layer_wise', exist_ok=True)

def analyze_layer(vanilla_weights, meta_weights, layer_name_vanilla, layer_name_meta, max_points=10000):
    """Analyze a single layer's weight space."""
    # Get weights
    v_weights = vanilla_weights[layer_name_vanilla].cpu().numpy()
    m_weights = meta_weights[layer_name_meta].cpu().numpy()
    
    # Skip if layer is too small for PCA
    if v_weights.size < 3:  # Need at least 3 points for meaningful 2D PCA
        return None, None, None
    
    # Reshape to 2D
    v_shape = v_weights.shape
    m_shape = m_weights.shape
    v_weights = v_weights.reshape(v_shape[0], -1)  # First dim is output channels
    m_weights = m_weights.reshape(m_shape[0], -1)
    
    # Sample points if too many
    if v_weights.shape[0] > max_points:
        indices = np.random.choice(v_weights.shape[0], max_points, replace=False)
        v_weights = v_weights[indices]
        m_weights = m_weights[indices]
    
    # Stack for PCA
    X = np.vstack([v_weights, m_weights])
    
    # Perform PCA
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X)
    
    # Split back
    n_points = len(X_transformed) // 2
    v_transformed = X_transformed[:n_points]
    m_transformed = X_transformed[n_points:]
    
    return v_transformed, m_transformed, pca.explained_variance_ratio_

def create_layer_plot(v_transformed, m_transformed, var_ratio, title):
    """Create comparison plot for a layer."""
    fig = make_subplots(rows=1, cols=2,
                       subplot_titles=['Vanilla Weights', 'Meta Weights'],
                       horizontal_spacing=0.1)
    
    # Add vanilla points
    fig.add_trace(
        go.Scatter(
            x=v_transformed[:, 0],
            y=v_transformed[:, 1],
            mode='markers',
            marker=dict(
                size=4,
                color='blue',
                opacity=0.6
            ),
            name='Vanilla'
        ),
        row=1, col=1
    )
    
    # Add meta points
    fig.add_trace(
        go.Scatter(
            x=m_transformed[:, 0],
            y=m_transformed[:, 1],
            mode='markers',
            marker=dict(
                size=4,
                color='red',
                opacity=0.6
            ),
            name='Meta'
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=f'{title}<br>PC1: {var_ratio[0]:.2%} var, PC2: {var_ratio[1]:.2%} var',
        width=1200,
        height=500,
        showlegend=True
    )
    
    # Ensure axes are aligned
    max_range = max(
        abs(v_transformed).max(),
        abs(m_transformed).max()
    )
    for i in [1, 2]:
        fig.update_xaxes(
            title='PC1',
            range=[-max_range, max_range],
            row=1, col=i
        )
        fig.update_yaxes(
            title='PC2',
            range=[-max_range, max_range],
            row=1, col=i
        )
    
    return fig

def create_progression_plot(layer_stats, arch):
    """Create a plot showing the progression of differences across layers."""
    # Extract data
    layers = [stat['layer'] for stat in layer_stats]
    mean_diffs = [stat['mean_diff'] for stat in layer_stats]
    percentile_95 = [stat['percentile_95'] for stat in layer_stats]
    var_ratios = [stat['var_ratio'][0] for stat in layer_stats]  # First PC explained variance
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add mean differences
    fig.add_trace(
        go.Scatter(
            x=layers,
            y=mean_diffs,
            name="Mean Difference",
            mode='lines+markers',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ),
        secondary_y=False
    )
    
    # Add 95th percentile differences
    fig.add_trace(
        go.Scatter(
            x=layers,
            y=percentile_95,
            name="95th Percentile Difference",
            mode='lines+markers',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=8)
        ),
        secondary_y=False
    )
    
    # Add explained variance ratio
    fig.add_trace(
        go.Scatter(
            x=layers,
            y=var_ratios,
            name="Explained Variance (PC1)",
            mode='lines+markers',
            line=dict(color='green', width=2),
            marker=dict(size=8)
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title=f'Layer-wise Progression for {arch}',
        xaxis=dict(
            title='Layer',
            tickangle=45
        ),
        yaxis=dict(
            title='Parameter Difference',
            gridcolor='lightgray'
        ),
        yaxis2=dict(
            title='Explained Variance Ratio',
            tickformat='.0%',
            gridcolor='lightgray'
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        width=1200,
        height=600,
        hovermode='x unified'
    )
    
    return fig

# Process each architecture
for arch, (vanilla_file, meta_file) in model_files.items():
    print(f"\nProcessing {arch}...")
    
    # Load weights
    vanilla_weights = load_model_weights(os.path.join(weights_dir, vanilla_file))
    meta_weights = load_model_weights(os.path.join(weights_dir, meta_file))
    
    # Get layer mappings
    if arch == 'conv6':
        layer_mappings = [
            ('conv1.weight', 'conv1.weight'),
            ('conv2.weight', 'conv2.weight'),
            ('conv3.weight', 'conv3.weight'),
            ('conv4.weight', 'conv4.weight'),
            ('conv5.weight', 'conv5.weight'),
            ('conv6.weight', 'conv6.weight'),
            ('fc1.weight', 'fc_layers.0.weight'),
            ('fc2.weight', 'fc_layers.1.weight'),
            ('fc3.weight', 'fc_layers.2.weight'),
            ('classifier.weight', 'classifier.weight')
        ]
    elif arch == 'conv4':
        layer_mappings = [
            ('conv1.weight', 'conv1.weight'),
            ('conv2.weight', 'conv2.weight'),
            ('conv3.weight', 'conv3.weight'),
            ('conv4.weight', 'conv4.weight'),
            ('fc1.weight', 'fc1.weight'),
            ('fc2.weight', 'fc2.weight'),
            ('fc3.weight', 'fc3.weight'),
            ('classifier.weight', 'classifier.weight')
        ]
    else:  # conv2
        layer_mappings = [
            ('conv1.weight', 'conv1.weight'),
            ('conv2.weight', 'conv2.weight'),
            ('fc1.weight', 'fc1.weight'),
            ('fc2.weight', 'fc2.weight'),
            ('fc3.weight', 'fc3.weight'),
            ('classifier.weight', 'classifier.weight')
        ]
    
    # Store layer statistics
    layer_stats = []
    
    print("\nAnalyzing layers:")
    for v_name, m_name in layer_mappings:
        print(f"Processing {v_name} -> {m_name}")
        
        # Get shapes for display
        v_shape = vanilla_weights[v_name].shape
        m_shape = meta_weights[m_name].shape
        print(f"Shapes: {v_shape} -> {m_shape}")
        
        # Analyze layer
        v_transformed, m_transformed, var_ratio = analyze_layer(
            vanilla_weights, meta_weights, v_name, m_name
        )
        
        # Skip if layer was too small
        if v_transformed is None:
            print("Layer too small for PCA analysis, skipping...")
            print()
            continue
        
        # Create and save individual layer plot
        fig = create_layer_plot(
            v_transformed, m_transformed, var_ratio,
            f"{arch} - {v_name}"
        )
        
        fig.write_image(f'pca_plots/layer_wise/{arch}_{v_name.replace(".", "_")}.png')
        print(f"Saved plot for {v_name}")
        
        # Calculate statistics
        diff = np.sqrt(np.sum((v_transformed - m_transformed)**2, axis=1))
        mean_diff = np.mean(diff)
        percentile_95 = np.percentile(diff, 95)
        
        # Store layer statistics
        layer_stats.append({
            'layer': v_name.split('.')[0],
            'mean_diff': mean_diff,
            'percentile_95': percentile_95,
            'var_ratio': var_ratio
        })
        
        print(f"Mean difference: {mean_diff:.4f}")
        print(f"95th percentile difference: {percentile_95:.4f}")
        print(f"Explained variance: {var_ratio[0]:.2%}, {var_ratio[1]:.2%}")
        print()
        
        # Free memory
        del v_transformed
        del m_transformed
        del fig
    
    # Create and save progression plot
    prog_fig = create_progression_plot(layer_stats, arch)
    prog_fig.write_image(f'pca_plots/layer_wise/{arch}_progression.png')
    print(f"Saved progression plot for {arch}")
    del prog_fig 