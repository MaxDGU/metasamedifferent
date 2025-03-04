import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from meta_baseline.weight_analysis import load_models, compare_weight_spaces
from meta_baseline.sparse_autoencoders import compare_models_with_autoencoders, collect_activations, analyze_activations
from meta_baseline.probing import compare_meta_and_baseline_representations
from meta_baseline.models.conv2lr import SameDifferentCNN as MetaCNN
from baselines.models.conv2 import SameDifferentCNN as BaselineCNN
from baselines.models.conv2 import SameDifferentDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze meta-learned vs baseline models')
    
    # Model paths
    parser.add_argument('--meta-model', type=str, required=True,
                        help='Path to meta-learned model checkpoint')
    parser.add_argument('--baseline-model', type=str, required=True,
                        help='Path to baseline model checkpoint')
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing dataset')
    parser.add_argument('--problem-number', type=int, required=True,
                        help='Problem number to analyze')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for processing')
    
    # Analysis options
    parser.add_argument('--analyze-weights', action='store_true',
                        help='Perform weight space analysis')
    parser.add_argument('--analyze-activations', action='store_true',
                        help='Perform activation analysis')
    parser.add_argument('--train-autoencoders', action='store_true',
                        help='Train and analyze with sparse autoencoders')
    parser.add_argument('--probe-representations', action='store_true',
                        help='Probe representations with linear classifiers')
    parser.add_argument('--run-all', action='store_true',
                        help='Run all analysis types')
    
    # Output directory
    parser.add_argument('--output-dir', type=str, default='model_analysis_results',
                        help='Directory to save analysis results')
    
    # Compute options
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cpu or cuda)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    args = parser.parse_args()
    
    # If run_all is specified, enable all analysis types
    if args.run_all:
        args.analyze_weights = True
        args.analyze_activations = True
        args.train_autoencoders = True
        args.probe_representations = True
    
    return args

def load_data(args):
    """Load data for analysis."""
    print(f"Loading data from {args.data_dir} for problem {args.problem_number}...")
    
    # Create datasets
    dataset = SameDifferentDataset(args.data_dir, args.problem_number, 'test')
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return dataloader

def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    print("Loading models...")
    meta_model, baseline_model = load_models(args.meta_model, args.baseline_model, device)
    
    # Load data
    dataloader = load_data(args)
    
    # Create output directories for each analysis type
    weight_dir = os.path.join(args.output_dir, 'weight_analysis')
    activation_dir = os.path.join(args.output_dir, 'activation_analysis')
    autoencoder_dir = os.path.join(args.output_dir, 'autoencoder_analysis')
    probe_dir = os.path.join(args.output_dir, 'probe_analysis')
    
    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(activation_dir, exist_ok=True)
    os.makedirs(autoencoder_dir, exist_ok=True)
    os.makedirs(probe_dir, exist_ok=True)
    
    # Run analyses based on arguments
    if args.analyze_weights:
        print("\n== Running Weight Space Analysis ==")
        compare_weight_spaces(meta_model, baseline_model, weight_dir)
    
    if args.analyze_activations:
        print("\n== Running Activation Analysis ==")
        # Define layers to analyze
        meta_layers = ['conv1', 'conv2', 'fc_layers.0', 'fc_layers.1', 'fc_layers.2']
        baseline_layers = ['conv1', 'conv2', 'fc1', 'fc2', 'fc3']
        
        # Collect and analyze meta-model activations
        print("Analyzing meta-model activations...")
        meta_activations = collect_activations(meta_model, dataloader, meta_layers, device)
        analyze_activations(meta_activations, os.path.join(activation_dir, 'meta'))
        
        # Collect and analyze baseline model activations
        print("Analyzing baseline model activations...")
        baseline_activations = collect_activations(baseline_model, dataloader, baseline_layers, device)
        analyze_activations(baseline_activations, os.path.join(activation_dir, 'baseline'))
    
    if args.train_autoencoders:
        print("\n== Running Sparse Autoencoder Analysis ==")
        compare_models_with_autoencoders(meta_model, baseline_model, dataloader, device)
    
    if args.probe_representations:
        print("\n== Running Representation Probing ==")
        # Collect labels for different probing tasks
        # Here we're using original labels as our probing target
        # In a more detailed analysis, you might derive multiple probe targets
        
        # Get all labels from the dataset
        all_labels = []
        for batch in dataloader:
            all_labels.append(batch['label'])
        all_labels = torch.cat(all_labels, dim=0)
        
        # Create probe tasks dictionary - can add more tasks here
        probe_tasks = {
            'same_different': all_labels  # Binary classification
        }
        
        # Run probing analysis
        compare_meta_and_baseline_representations(
            meta_model, baseline_model, dataloader, 
            probe_tasks, probe_dir, device
        )
    
    print(f"\nAll analyses complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 