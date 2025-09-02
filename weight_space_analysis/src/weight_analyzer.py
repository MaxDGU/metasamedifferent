import os
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd

class WeightSpaceAnalyzer:
    """Analyzes the weight space of CNN models."""
    
    def __init__(self, model_type: str, architecture: str):
        """
        Initialize the analyzer.
        
        Args:
            model_type: Either 'vanilla' or 'meta'
            architecture: One of 'conv2', 'conv4', 'conv6'
        """
        if model_type not in ['vanilla', 'meta']:
            raise ValueError("model_type must be either 'vanilla' or 'meta'")
        if architecture not in ['conv2', 'conv4', 'conv6']:
            raise ValueError("architecture must be one of: 'conv2', 'conv4', 'conv6'")
            
        self.model_type = model_type
        self.architecture = architecture
        self.weights_data = {}
        
    def load_weights(self, base_path: str, seeds: List[int]) -> None:
        """
        Load weights for all seeds.
        
        Args:
            base_path: Base path to model weights
            seeds: List of seed values to load
        """
        for seed in seeds:
            if self.model_type == 'meta':
                path = os.path.join(base_path, 'meta_baselines', 
                                  self.architecture, f'seed_{seed}', 
                                  'best_model.pt')
            else:
                path = os.path.join(base_path, 'results/pb_baselines/all_tasks',
                                  self.architecture, f'seed_{seed}',
                                  'best_model.pt')
            
            try:
                state_dict = torch.load(path, map_location='cpu')
                if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                self.weights_data[seed] = state_dict
            except Exception as e:
                print(f"Error loading weights for seed {seed}: {str(e)}")
    
    def extract_layer_weights(self, layer_name: str) -> Dict[int, np.ndarray]:
        """
        Extract weights from a specific layer across all seeds.
        
        Args:
            layer_name: Name of the layer to extract (e.g., 'conv1.weight')
            
        Returns:
            Dictionary mapping seeds to weight arrays
        """
        weights = {}
        for seed, state_dict in self.weights_data.items():
            if layer_name in state_dict:
                # Convert to numpy and reshape
                w = state_dict[layer_name].cpu().numpy()
                # For conv layers: (out_channels, in_channels, kernel_h, kernel_w)
                # Reshape to 2D: (out_channels, in_channels * kernel_h * kernel_w)
                if len(w.shape) == 4:
                    w = w.reshape(w.shape[0], -1)
                weights[seed] = w
        return weights
    
    def compute_statistics(self, weights: np.ndarray) -> Dict[str, float]:
        """
        Compute basic statistics for weight values.
        
        Args:
            weights: numpy array of weights
            
        Returns:
            Dictionary of statistics
        """
        return {
            'mean': float(np.mean(weights)),
            'std': float(np.std(weights)),
            'min': float(np.min(weights)),
            'max': float(np.max(weights)),
            'l2_norm': float(np.linalg.norm(weights)),
            'sparsity': float(np.mean(np.abs(weights) < 1e-6))
        }
    
    def reduce_dimensions(self, weights: np.ndarray, 
                         method: str = 'pca',
                         n_components: int = 2) -> np.ndarray:
        """
        Reduce dimensionality of weight vectors.
        
        Args:
            weights: numpy array of shape (n_filters, n_features)
            method: One of 'pca', 'tsne', or 'umap'
            n_components: Number of components in reduced space
            
        Returns:
            Reduced weights of shape (n_filters, n_components)
        """
        if method == 'pca':
            reducer = PCA(n_components=n_components)
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42)
        elif method == 'umap':
            reducer = umap.UMAP(n_components=n_components, random_state=42)
        else:
            raise ValueError("method must be one of: 'pca', 'tsne', 'umap'")
            
        return reducer.fit_transform(weights)


class WeightSpaceVisualizer:
    """Creates interactive visualizations of CNN weight spaces."""
    
    def __init__(self):
        """Initialize the visualizer."""
        pass
    
    def create_distribution_plot(self, 
                               vanilla_weights: Dict[int, np.ndarray],
                               meta_weights: Dict[int, np.ndarray],
                               layer_name: str) -> go.Figure:
        """
        Create distribution plots comparing vanilla and meta-learned weights.
        
        Args:
            vanilla_weights: Dictionary mapping seeds to weight arrays for vanilla model
            meta_weights: Dictionary mapping seeds to weight arrays for meta-learned model
            layer_name: Name of the layer being visualized
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Add traces for vanilla weights
        for seed, weights in vanilla_weights.items():
            fig.add_trace(go.Violin(
                x=['vanilla'] * len(weights.flatten()),
                y=weights.flatten(),
                name=f'Vanilla (seed {seed})',
                side='negative',
                line_color='blue'
            ))
        
        # Add traces for meta weights
        for seed, weights in meta_weights.items():
            fig.add_trace(go.Violin(
                x=['meta'] * len(weights.flatten()),
                y=weights.flatten(),
                name=f'Meta (seed {seed})',
                side='positive',
                line_color='red'
            ))
        
        fig.update_layout(
            title=f'Weight Distribution: {layer_name}',
            xaxis_title='Model Type',
            yaxis_title='Weight Value',
            violinmode='overlay',
            showlegend=True
        )
        
        return fig
    
    def create_embedding_plot(self,
                            vanilla_embeddings: Dict[int, np.ndarray],
                            meta_embeddings: Dict[int, np.ndarray],
                            layer_name: str,
                            method: str) -> go.Figure:
        """
        Create scatter plot of reduced-dimension weight embeddings.
        
        Args:
            vanilla_embeddings: Dictionary mapping seeds to embeddings for vanilla model
            meta_embeddings: Dictionary mapping seeds to embeddings for meta-learned model
            layer_name: Name of the layer being visualized
            method: Reduction method used ('pca', 'tsne', or 'umap')
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Add traces for vanilla embeddings
        for seed, emb in vanilla_embeddings.items():
            fig.add_trace(go.Scatter(
                x=emb[:, 0],
                y=emb[:, 1],
                mode='markers',
                name=f'Vanilla (seed {seed})',
                marker=dict(size=8, opacity=0.6),
                hovertemplate=(
                    'Seed: %{text}<br>' +
                    'x: %{x:.2f}<br>' +
                    'y: %{y:.2f}<br>'
                ),
                text=[f'{seed}'] * len(emb)
            ))
        
        # Add traces for meta embeddings
        for seed, emb in meta_embeddings.items():
            fig.add_trace(go.Scatter(
                x=emb[:, 0],
                y=emb[:, 1],
                mode='markers',
                name=f'Meta (seed {seed})',
                marker=dict(size=8, opacity=0.6),
                hovertemplate=(
                    'Seed: %{text}<br>' +
                    'x: %{x:.2f}<br>' +
                    'y: %{y:.2f}<br>'
                ),
                text=[f'{seed}'] * len(emb)
            ))
        
        fig.update_layout(
            title=f'{method.upper()} Embedding: {layer_name}',
            xaxis_title=f'{method.upper()} 1',
            yaxis_title=f'{method.upper()} 2',
            showlegend=True
        )
        
        return fig 