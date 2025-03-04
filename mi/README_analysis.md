# Neural Network Weight Space Analysis Tools

This collection of tools is designed to analyze and compare the weight spaces and internal representations of meta-learned versus non-meta-learned convolutional neural networks.

## Overview

The toolset provides several different methods for mechanistic interpretability and analysis:

1. **Weight Space Analysis**: Direct comparison of neural network weight matrices through PCA visualization and statistical analysis.

2. **Activation Analysis**: Visualization and analysis of neural activations across different layers of the models.

3. **Sparse Autoencoders**: Implementation of sparse autoencoders to extract interpretable features from model activations.

4. **Linear Probing**: Training linear classifiers to probe what information is encoded at each layer of the networks.

## Project Structure

- **weight_analysis.py**: Functions for analyzing and visualizing weight matrices using PCA and other techniques.
- **sparse_autoencoders.py**: Implementation of sparse autoencoders for feature extraction and visualization.
- **probing.py**: Linear and non-linear probing functions to examine what information is represented in each layer.
- **analyze_models.py**: Main script that brings together all analysis tools with a unified interface.

## Requirements

- PyTorch
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn
- tqdm

## Usage

### Basic Usage

The primary entry point is the `analyze_models.py` script, which can be run with various options:

```bash
python -m meta_baseline.analyze_models \
    --meta-model path/to/meta_model.pt \
    --baseline-model path/to/baseline_model.pt \
    --data-dir path/to/data \
    --problem-number 1 \
    --run-all
```

### Command-line Arguments

- `--meta-model`: Path to the meta-learned model checkpoint
- `--baseline-model`: Path to the baseline model checkpoint
- `--data-dir`: Directory containing dataset
- `--problem-number`: Problem number to analyze
- `--batch-size`: Batch size for processing (default: 32)

Analysis Options:
- `--analyze-weights`: Perform weight space analysis using PCA
- `--analyze-activations`: Analyze activations across layers
- `--train-autoencoders`: Train sparse autoencoders on activations
- `--probe-representations`: Use linear and non-linear probes to analyze representations
- `--run-all`: Run all analysis types

Output Options:
- `--output-dir`: Directory to save analysis results (default: 'model_analysis_results')

Compute Options:
- `--device`: Device to use ('cpu' or 'cuda', default: 'cuda')
- `--num-workers`: Number of data loading workers (default: 4)

## Interpreting the Results

### Weight Space Analysis

The weight space analysis produces PCA visualizations that show how the weight matrices of different layers are distributed in a 2D projection. Differences between meta-learned and baseline models may reveal:

- Different clustering of weights
- Different degrees of sparsity
- Different magnitudes and distributions of weights

### Activation Analysis

Activation analysis examines the distribution of activations at each layer, which can reveal:

- How information flows through the network
- Which layers extract what type of features
- How representations differ between meta-learned and non-meta-learned models

### Sparse Autoencoders

Sparse autoencoders extract interpretable features from the activations. Analysis of these features can reveal:

- What features are detected by each layer
- How features combine across layers
- Differences in feature extraction between meta-learned and baseline models

### Linear Probing

Probing results show how decodable certain information is from each layer. This helps understand:

- What information is explicitly represented at each layer
- How representations evolve through the network
- Whether meta-learning produces more generalizable or task-specific representations

## Example Workflow

1. **Train models**: Ensure you have both meta-learned and baseline models trained.
2. **Run weight analysis**: Start with weight analysis to understand structural differences.
3. **Analyze activations**: Run activation analysis to see how inputs are processed.
4. **Train autoencoders**: Extract interpretable features with sparse autoencoders.
5. **Probe representations**: Run linear probes to test what information is encoded in each layer.
6. **Compare results**: Look for patterns in the differences between meta-learned and non-meta-learned models.

## Advanced Usage

For more detailed analysis, you can use the individual modules directly in your own Python scripts:

```python
from meta_baseline.weight_analysis import load_models, compare_weight_spaces
from meta_baseline.sparse_autoencoders import collect_activations, train_sparse_autoencoder
from meta_baseline.probing import probe_layer_representations

# Load models
meta_model, baseline_model = load_models('meta_model.pt', 'baseline_model.pt')

# Perform custom analysis
# ...
```

## References and Further Reading

For more information on the techniques used in this toolset:

1. Sparse Autoencoders for Interpretable Feature Learning
   - Anthropic's Sparse Autoencoders: https://transformer-circuits.pub/2023/monosemantic-features

2. Linear Probing for Model Interpretability
   - Alain & Bengio, "Understanding intermediate layers using linear classifier probes" (2016)

3. Dimensionality Reduction and Visualization
   - PCA for weights visualization: https://distill.pub/2016/misread-tsne/ 