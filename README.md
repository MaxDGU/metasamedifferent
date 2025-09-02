# Meta-Learning for Same-Different Visual Classification

Code and experiments comparing Model-Agnostic Meta-Learning (MAML) against standard supervised learning on same-different visual classification tasks.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Generate datasets (requires ~100GB storage)
python data/meta_data_generator_h5.py      # Meta-learning datasets
python data/vanilla_h5_dataset_creation.py # Vanilla training datasets
```

For cluster runs, activate the conda environment:
```bash
conda activate tensorflow
```

## Experiments

### 1. Baseline (Vanilla SGD)

Standard supervised learning on Puebla-Bowers (PB) shape tasks.

```bash
# Local
python baselines/run_pb_baselines.py --architecture conv6 --seed 42

# Cluster  
sbatch slurm/pb/run_pb_conv6_training.slurm
```

**Output:** `results/pb_baselines/`

### 2. Meta-Learning Baseline

MAML training on the same task distribution as baselines.

```bash
# Local
python meta_baseline/train_and_test_meta_baselines.py \
    --architecture conv6 --seed 42 \
    --adaptation_steps 5 --test_adaptation_steps 15

# Cluster
sbatch slurm/meta/run_fomaml_array_della.slurm
```

**Output:** `results/meta_baselines/`

### 3. Holdout (Out-of-Distribution)

Meta-learning generalization to unseen tasks.

```bash
# Local
python holdout_experiment/experiment1_leave_one_out.py \
    --held_out_task filled --architecture conv6 --seed 42

# Cluster
sbatch slurm/holdout_experiment/run_experiment1.slurm
```

**Output:** `results/holdout_experiment/`

### 4. Naturalistic Images

Training and testing on real-world naturalistic images.

**Meta-learning:**
```bash
python naturalistic/meta_naturalistic_train.py --seed 42
```

**Vanilla training:**
```bash
python naturalistic/train_vanilla_new_arch.py --seed 42
```

**Cross-domain testing (PB → Naturalistic):**
```bash
python naturalistic/test_naturalistic_meta.py     # Meta models
python naturalistic/test_naturalistic_vanilla.py  # Vanilla models
```

**Output:** `results/naturalistic/`

### 5. Weight Space Analysis

PCA analysis of learned representations.

```bash
python weight_space_analysis/analyze_all_weights.py  # Main analysis
python scripts/visualize_meta_vs_vanilla_pca.py      # Visualizations
python plot_pca.py                                   # Simple plot
```

**Output:** `weight_space_analysis/`, `visualizations/`

### 6. Sample Efficiency

Compare training efficiency between methods.

```bash
sbatch slurm/sample_efficiency/run_sample_efficiency_comparison.slurm
python scripts/sample_efficiency_comparison.py
```

## Results

Key result files after running experiments:

- **Baselines:** `results/pb_baselines/compiled_results.json`
- **Meta-learning:** `results/meta_baselines/`
- **Holdout:** `results/holdout_experiment/`
- **Naturalistic:** `results/naturalistic/`
- **Weight analysis:** `weight_space_analysis/`, `visualizations/`

Generate paper figures:
```bash
python plotting/plot_pb_baselines.py        # Baseline results
python plotting/plot_experiment1_results.py # Experimental comparisons
python plot_pca.py                          # PCA plots
```

## Model Architectures

Three CNN architectures: **conv2** (2-layer), **conv4** (4-layer), **conv6** (6-layer)

- **Vanilla:** `baselines/models/conv{2,4,6}.py`
- **Meta-learning:** `meta_baseline/models/conv{2,4,6}lr.py`

## Configuration

**MAML parameters:**
- Inner LR: 0.05, Outer LR: 0.001
- Adaptation steps: 5 (train), 15 (test)
- Support sizes: [4,6,8,10], Query size: 3

**Vanilla SGD:**
- LR: 0.001, Batch size: 32, Patience: 10

## Repository Structure

```
metasamedifferent/
├── data/                    # Dataset generation
├── baselines/               # Vanilla SGD experiments
├── meta_baseline/           # MAML experiments
├── holdout_experiment/      # OOD generalization
├── naturalistic/            # Real image experiments
├── weight_space_analysis/   # Weight analysis
├── scripts/                 # Analysis scripts
├── slurm/                   # Cluster job scripts
├── results/                 # Experimental outputs
└── plotting/                # Visualization scripts
```

## Hardware Requirements

- **GPU:** CUDA-compatible (recommended)
- **RAM:** 32GB+ for large experiments
- **Storage:** ~100GB for datasets and results

## Troubleshooting

**Common issues:**
- CUDA OOM → reduce batch size
- Missing datasets → run data generation scripts first
- Import errors → ensure project root in Python path

**Cluster setup:**
```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
```
