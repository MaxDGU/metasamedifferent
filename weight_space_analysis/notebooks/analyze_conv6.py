# %% [markdown]
# # Model Weight Space Analysis
# 
# This notebook analyzes and visualizes the weight spaces of different CNN models (Conv2, Conv4, Conv6), comparing vanilla and meta-learned variants.

# %% [markdown]
# ## Setup
# First, let's import our dependencies and setup our analyzers

# %%
import sys
sys.path.append('../src')

import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
import plotly.io as pio
pio.templates.default = 'plotly_white'
from sklearn.decomposition import PCA
import os
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# For better notebook display
import warnings
warnings.filterwarnings('ignore')

# Check environment and kaleido installation
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
try:
    import kaleido
    print(f"Kaleido version: {kaleido.__version__}")
    print(f"Kaleido location: {kaleido.__file__}")
except ImportError as e:
    print(f"Kaleido import error: {e}")

# %% [markdown]
# ## Load Model Weights
# We'll load the weights from all model files: conv2, conv4, and conv6 for both vanilla and meta variants

# %%
def load_model_weights(filepath):
    """Load weights from a model file."""
    state_dict = torch.load(filepath, map_location='cpu')
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        return state_dict['model_state_dict']
    return state_dict

def get_layer_statistics(param):
    """Compute statistics for a layer's parameters."""
    values = param.cpu().numpy().flatten()
    return np.array([
        np.mean(values),
        np.std(values),
        np.median(values),
        np.percentile(values, 25),
        np.percentile(values, 75),
        np.min(values),
        np.max(values),
        np.mean(np.abs(values)),
        np.std(np.abs(values))
    ])

def get_layer_features(state_dict):
    """Compute feature vectors for each layer."""
    layer_features = {}
    for name, param in state_dict.items():
        if 'weight' in name:  # Only process weight parameters
            layer_features[name] = get_layer_statistics(param)
    return layer_features

# Load all models
weights_dir = '../weights'
model_files = {
    'conv2': ('best_model_vanilla2.pt', 'best_model_meta2.pt'),
    'conv4': ('best_model_vanilla4.pt', 'best_model_meta4.pt'),
    'conv6': ('best_model_vanilla6.pt', 'best_model_meta6.pt')
}

# Create output directory
os.makedirs('pca_plots', exist_ok=True)

# Collect layer features for all models
all_layer_features = []
layer_labels = []
model_types = []
architectures = []

# Process each model
for arch, (vanilla_file, meta_file) in model_files.items():
    # Load weights
    vanilla_weights = load_model_weights(os.path.join(weights_dir, vanilla_file))
    meta_weights = load_model_weights(os.path.join(weights_dir, meta_file))
    
    # Get layer features
    vanilla_features = get_layer_features(vanilla_weights)
    meta_features = get_layer_features(meta_weights)
    
    # Add vanilla layer features
    for name, features in vanilla_features.items():
        all_layer_features.append(features)
        layer_labels.append(name)
        model_types.append('vanilla')
        architectures.append(arch)
    
    # Add meta layer features
    for name, features in meta_features.items():
        all_layer_features.append(features)
        layer_labels.append(name)
        model_types.append('meta')
        architectures.append(arch)

# Convert to numpy array
X = np.array(all_layer_features)

# Standardize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Perform PCA
pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X)

# Create visualization
    fig = go.Figure()
    
# Define colors for each model type (6 distinct colors)
colors = {
    'conv2_vanilla': '#1f77b4',  # blue
    'conv2_meta': '#ff7f0e',     # orange
    'conv4_vanilla': '#2ca02c',  # green
    'conv4_meta': '#d62728',     # red
    'conv6_vanilla': '#9467bd',  # purple
    'conv6_meta': '#8c564b'      # brown
}

# Group points by model type
for arch in ['conv2', 'conv4', 'conv6']:
    for model_type in ['vanilla', 'meta']:
        model_id = f"{arch}_{model_type}"
        
        # Get indices for this model type
        indices = [i for i, (a, t) in enumerate(zip(architectures, model_types)) 
                  if a == arch and t == model_type]
        
        # Add all points for this model type
        fig.add_trace(go.Scatter(
            x=X_transformed[indices, 0],
            y=X_transformed[indices, 1],
            mode='markers',
            marker=dict(
                size=8,
                color=colors[model_id],
                symbol='circle',
                line=dict(width=1, color='black')
            ),
            name=model_id,
            hovertemplate="Layer: %{customdata}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
            customdata=[layer_labels[i] for i in indices],
            showlegend=True
        ))
        
        # Add arrows between corresponding vanilla and meta layers if this is meta
        if model_type == 'meta':
            for idx in indices:
                layer = layer_labels[idx]
                # Find corresponding vanilla layer
                vanilla_idx = None
                for j, (l, t, a) in enumerate(zip(layer_labels, model_types, architectures)):
                    if l == layer and t == 'vanilla' and a == arch:
                        vanilla_idx = j
                        break
                
                if vanilla_idx is not None:
                    fig.add_trace(go.Scatter(
                        x=[X_transformed[vanilla_idx, 0], X_transformed[idx, 0]],
                        y=[X_transformed[vanilla_idx, 1], X_transformed[idx, 1]],
                        mode='lines',
                        line=dict(color='rgba(128, 128, 128, 0.3)', dash='dot', width=1),
                        showlegend=False,
                        hoverinfo='skip'
    ))
    
    # Update layout
    fig.update_layout(
    title=f'PCA of Layer-wise Features<br>PC1: {pca.explained_variance_ratio_[0]:.2%} var, PC2: {pca.explained_variance_ratio_[1]:.2%} var',
    width=1500,
    height=1000,
    xaxis_title="PC1",
    yaxis_title="PC2",
    showlegend=True,
    legend=dict(
        title="Model Type",
        groupclick="toggleitem",
        itemsizing="constant"
    )
)

# Add grid
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

# Save plot
fig.write_image('pca_plots/layer_wise_pca.png')
print("Saved PCA plot of layer-wise features")
print(f"Explained variance ratios: PC1={pca.explained_variance_ratio_[0]:.4f}, PC2={pca.explained_variance_ratio_[1]:.4f}")

# Print feature importance for each principal component
feature_names = ['mean', 'std', 'median', 'q25', 'q75', 'min', 'max', 'abs_mean', 'abs_std']
print("\nFeature importance:")
for i, pc in enumerate(pca.components_[:2]):
    print(f"\nPC{i+1} contributions:")
    for fname, importance in zip(feature_names, pc):
        print(f"{fname}: {importance:.4f}")

# Print average distances between vanilla and meta layers for each architecture and layer type
for arch in ['conv2', 'conv4', 'conv6']:
    print(f"\nAnalyzing {arch} layer differences:")
    
    # Get indices for this architecture
    arch_indices = [i for i, a in enumerate(architectures) if a == arch]
    vanilla_indices = [i for i in arch_indices if model_types[i] == 'vanilla']
    meta_indices = [i for i in arch_indices if model_types[i] == 'meta']
    
    # Calculate distances for matching layers
    layer_type_distances = {}
    for v_idx, m_idx in zip(vanilla_indices, meta_indices):
        if layer_labels[v_idx] == layer_labels[m_idx]:  # Ensure we're comparing same layers
            v_point = X_transformed[v_idx]
            m_point = X_transformed[m_idx]
            distance = np.linalg.norm(m_point - v_point)
            
            layer_type = layer_labels[v_idx].split('.')[0]
            if layer_type not in layer_type_distances:
                layer_type_distances[layer_type] = []
            layer_type_distances[layer_type].append(distance)
            
            print(f"{layer_labels[v_idx]}: {distance:.4f}")
    
    print("\nAverage distances by layer type:")
    for layer_type, distances in layer_type_distances.items():
        print(f"{layer_type}: {np.mean(distances):.4f}")

def create_density_plot(X, title, colormap):
    try:
        # For very small layers, use scatter plot instead
        if X.shape[0] * X.shape[1] < 5000:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
                x=X[0], y=X[1],
        mode='markers',
        marker=dict(
            color='blue',
                    size=3,
                    opacity=0.5
                ),
                name=title
            ))
        else:
            # Try kernel density estimation
            try:
                kernel = gaussian_kde(X.T)
            except np.linalg.LinAlgError:
                # If KDE fails, fall back to 2D histogram
                fig = go.Figure()
                fig.add_trace(go.Histogram2d(
                    x=X[0], y=X[1],
                    colorscale=colormap,
                    nbinsx=50,
                    nbinsy=50,
                    name=title
                ))
                return fig
            
            # Create grid of points
            x_grid = np.linspace(X[0].min(), X[0].max(), 100)
            y_grid = np.linspace(X[1].min(), X[1].max(), 100)
            xx, yy = np.meshgrid(x_grid, y_grid)
            positions = np.vstack([xx.ravel(), yy.ravel()])
            
            # Evaluate kernel on grid
            z = np.reshape(kernel(positions).T, xx.shape)
            
            # Create contour plot
            fig = go.Figure()
            fig.add_trace(go.Contour(
                x=x_grid,
                y=y_grid,
                z=z,
                colorscale=colormap,
                contours=dict(
                    start=0,
                    end=z.max(),
                    size=(z.max() - 0) / 10
                ),
                name=title
    ))
    
    # Update layout
    fig.update_layout(
            title=title,
            xaxis_title="PC1",
            yaxis_title="PC2",
            showlegend=False
    )
    
    return fig

    except Exception as e:
        print(f"Error creating density plot for {title}: {str(e)}")
        # Return empty figure as fallback
        return go.Figure()

def create_comparison_plot(vanilla_transformed, meta_transformed, var_ratio, title):
    """Create a three-panel comparison plot."""
    # Create subplots
    fig = make_subplots(rows=1, cols=3, 
                       subplot_titles=('Vanilla Weights', 'Meta Weights', 'Weight Differences'),
                       horizontal_spacing=0.1)
    
    # Get vanilla and meta density plots
    vanilla_fig = create_density_plot(vanilla_transformed.T, 'Vanilla', 'Blues')
    meta_fig = create_density_plot(meta_transformed.T, 'Meta', 'Reds')
    
    # Add vanilla plot
    for trace in vanilla_fig.data:
        fig.add_trace(trace, row=1, col=1)
    
    # Add meta plot
    for trace in meta_fig.data:
        fig.add_trace(trace, row=1, col=2)
    
    # Calculate and plot differences
    differences = meta_transformed - vanilla_transformed
    diff_fig = go.Figure()
    diff_fig.add_trace(go.Scatter(
        x=differences[0],
        y=differences[1],
        mode='markers',
        marker=dict(
            color=np.sqrt(differences[0]**2 + differences[1]**2),
            colorscale='RdBu',
            size=3,
            colorbar=dict(title='Magnitude of Change'),
            cmin=-np.abs(differences).max(),
            cmax=np.abs(differences).max()
        ),
        name='Differences'
    ))
    
    # Add difference plot
    for trace in diff_fig.data:
        fig.add_trace(trace, row=1, col=3)
    
    # Update layout
    fig.update_layout(
        title=f"{title} - Explained Variance: {var_ratio[0]:.2%}, {var_ratio[1]:.2%}",
        width=1800,
        height=600,
        showlegend=False
    )
    
    # Update axes
    for i in range(1, 4):
        fig.update_xaxes(title_text="PC1", row=1, col=i, gridcolor='lightgray', showgrid=True)
        fig.update_yaxes(title_text="PC2", row=1, col=i, gridcolor='lightgray', showgrid=True)
    
    return fig

# Weights directory and model files setup
weights_dir = '../weights'
model_files = {
    'conv2': ('best_model_vanilla2.pt', 'best_model_meta2.pt'),
    'conv4': ('best_model_vanilla4.pt', 'best_model_meta4.pt'),
    'conv6': ('best_model_vanilla6.pt', 'best_model_meta6.pt')
}

# Create output directory
os.makedirs('pca_plots', exist_ok=True)

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
    
    print("\nAnalyzing layers:")
    for v_name, m_name in layer_mappings:
        print(f"Processing {v_name} -> {m_name}")
        
        # Get shapes for display
        v_shape = vanilla_weights[v_name].shape
        m_shape = meta_weights[m_name].shape
        print(f"Shapes: {v_shape} -> {m_shape}")
        
        # Get weights and reshape
        v_weights = vanilla_weights[v_name].cpu().numpy().reshape(v_shape[0], -1)
        m_weights = meta_weights[m_name].cpu().numpy().reshape(m_shape[0], -1)
        
        # Stack for PCA
        X = np.vstack([v_weights, m_weights])
        
        # Perform PCA
        pca = PCA(n_components=2)
        X_transformed = pca.fit_transform(X)
        
        # Split back
        n_points = len(X_transformed) // 2
        v_transformed = X_transformed[:n_points]
        m_transformed = X_transformed[n_points:]
        
        # Create and save plot
        fig = create_comparison_plot(
            v_transformed, m_transformed,
            pca.explained_variance_ratio_,
            f"{arch} - {v_name}"
        )
        
        fig.write_image(f'pca_plots/{arch}_{v_name.replace(".", "_")}.png')
        print(f"Saved plot for {v_name}")
        
        # Calculate statistics
        diff = np.sqrt(np.sum((v_transformed - m_transformed)**2, axis=1))
        print(f"Mean difference: {np.mean(diff):.4f}")
        print(f"95th percentile difference: {np.percentile(diff, 95):.4f}")
        print(f"Explained variance: {pca.explained_variance_ratio_[0]:.2%}, {pca.explained_variance_ratio_[1]:.2%}")
        print()
        
        # Free memory
        del v_transformed
        del m_transformed
        del fig 