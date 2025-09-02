#!/usr/bin/env python
# naturalistic/test_vanilla_model.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import random
import sys
import gc
import json
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from baselines.models.conv2 import SameDifferentCNN as Conv2CNN
    from baselines.models.conv4 import SameDifferentCNN as Conv4CNN
    from baselines.models.conv6 import SameDifferentCNN as Conv6CNN
    print("Successfully imported Conv{2,4,6}lrCNN models from baselines.models")
except ImportError as e:
    print(f"Error importing models: {e}")
    print("Please ensure conv2.py, conv4.py, conv6.py are in baselines/models/ and project root is in PYTHONPATH.")
    exit(1)

# Configuration for models and seeds to test
ARCHITECTURES_CONFIG = {
    'conv2lr': {'class': Conv2CNN, 'seeds': [42, 123, 789, 555, 999]},
    'conv4lr': {'class': Conv4CNN, 'seeds': [42, 123, 789, 555, 999]},
    'conv6lr': {'class': Conv6CNN, 'seeds': [42, 123, 789, 555, 999]}
}

# Copied from train_vanilla.py, ensure it's consistent
class NaturalisticDataset(Dataset):
    """Dataset for naturalistic same/different classification."""
    def __init__(self, root_dir, split='test', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        same_dir = self.root_dir / split / 'same'
        diff_dir = self.root_dir / split / 'different'
        if not same_dir.exists() or not diff_dir.exists():
            raise FileNotFoundError(f"Data directory for split '{split}' not found or incomplete in {self.root_dir}")
        same_files = list(same_dir.glob('*.png'))
        diff_files = list(diff_dir.glob('*.png'))
        self.file_paths = same_files + diff_files
        self.labels = ([1] * len(same_files)) + ([0] * len(diff_files)) # 1 for same, 0 for different
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
    def __len__(self):
        return len(self.file_paths)
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return {'image': image, 'label': label}

# Adapted from validate function in train_vanilla.py
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0
    pbar = tqdm(test_loader, desc='Testing', leave=False)
    with torch.no_grad():
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            running_loss += loss.item() * images.size(0)
            running_acc += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            pbar.set_postfix({
                'loss': running_loss / total_samples if total_samples > 0 else 0,
                'acc': running_acc / total_samples if total_samples > 0 else 0
            })
    avg_loss = running_loss / total_samples if total_samples > 0 else 0
    avg_acc = running_acc / total_samples if total_samples > 0 else 0
    return avg_loss, avg_acc

def parse_args():
    parser = argparse.ArgumentParser(description="Test trained Vanilla CNN models, aggregate results, and plot.")
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory of the naturalistic dataset (e.g., trainsize_6400_1200-300-100), must contain a split.')
    parser.add_argument('--base_log_dir', type=str, required=True, help='Base directory where trained vanilla models are stored (e.g., logs_naturalistic_vanilla)')
    parser.add_argument('--output_json_path', type=str, default='test_results_vanilla/vanilla_summary.json', help='Path to save aggregated results in JSON.')
    parser.add_argument('--output_plot_path', type=str, default='test_results_vanilla/vanilla_summary_plot.png', help='Path to save the summary plot.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing.')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for DataLoader.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use.')
    return parser.parse_args()

# Custom JSON encoder to handle numpy types that might be in args from checkpoint
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, Path): return str(obj)
        return super(NpEncoder, self).default(obj)

def run_single_vanilla_test(arch_name, seed, model_class, args):
    device = torch.device(args.device)
    model_dir = Path(args.base_log_dir) / arch_name / f"seed_{seed}"
    checkpoint_path = model_dir / "best_model.pt"

    print(f"\nAttempting to test Vanilla Model: Arch={arch_name}, Seed={seed}")
    print(f"Looking for model at: {checkpoint_path}")

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}. Skipping.")
        return None

    model = model_class().to(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'args' in checkpoint:
            train_args = checkpoint['args']
            print(f"  Successfully loaded. Model trained with args: {train_args}")
        else:
            print("  Successfully loaded. No 'args' in checkpoint.")
    except Exception as e:
        print(f"Error loading checkpoint for {arch_name} seed {seed}: {e}. Skipping.")
        return None

    # Data loading
    try:
        test_dataset = NaturalisticDataset(args.data_dir, split='test')
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        print(f"  Testing on {len(test_dataset)} images from {Path(args.data_dir) / 'test'}")
    except Exception as e:
        print(f"Error loading test dataset: {e}. Skipping.")
        return None

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

    print(f"  Test Result for {arch_name} seed {seed}: Avg Loss: {test_loss:.4f}, Avg Acc: {test_acc:.4f}")
    gc.collect()
    return test_acc

def main():
    args = parse_args()
    Path(args.output_json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_plot_path).parent.mkdir(parents=True, exist_ok=True)

    all_results = {}
    print(f"Starting vanilla model testing. Results will be saved to parent directory of {args.output_json_path}")
    print(f"Reading models from base: {args.base_log_dir}")
    print(f"Reading test data from: {args.data_dir}")

    for arch_name, config in ARCHITECTURES_CONFIG.items():
        model_class = config['class']
        seeds = config['seeds']
        arch_accuracies = []
        print(f"\n--- Processing Architecture: {arch_name} ---")
        for seed in seeds:
            # Set seed for consistency if any torch operations during test had randomness (though should be minimal for eval)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

            test_acc = run_single_vanilla_test(arch_name, seed, model_class, args)
            if test_acc is not None:
                arch_accuracies.append(test_acc)
        
        if arch_accuracies:
            mean_acc = np.mean(arch_accuracies)
            std_acc = np.std(arch_accuracies)
            all_results[arch_name] = {
                "accuracies": arch_accuracies,
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "num_seeds_successful": len(arch_accuracies)
            }
            print(f"Summary for {arch_name}: Mean Acc: {mean_acc:.4f} +/- {std_acc:.4f} (from {len(arch_accuracies)} seeds)")
        else:
            all_results[arch_name] = {"accuracies": [], "mean_accuracy": 0.0, "std_accuracy": 0.0, "num_seeds_successful": 0}
            print(f"No successful test runs for architecture {arch_name}")

    print(f"\nSaving aggregated vanilla test results to {args.output_json_path}")
    with open(args.output_json_path, 'w') as f:
        json.dump(all_results, f, indent=4, cls=NpEncoder)

    arch_names_plot = list(all_results.keys())
    mean_accuracies_plot = [all_results[arch].get('mean_accuracy', 0) for arch in arch_names_plot]
    std_accuracies_plot = [all_results[arch].get('std_accuracy', 0) for arch in arch_names_plot]

    if not any(mean_accuracies_plot):
        print("No data to plot or all accuracies are zero. Skipping plot generation.")
    else:
        plt.figure(figsize=(10, 6))
        plt.bar(arch_names_plot, mean_accuracies_plot, yerr=std_accuracies_plot, capsize=5, color=['#ff9999','#66b3ff','#99ff99']) # Different colors
        plt.xlabel("Model Architecture (Vanilla SGD)")
        plt.ylabel("Mean Test Accuracy")
        plt.title("Mean Test Accuracy by Architecture (Vanilla SGD)")
        plt.ylim(0, 1)
        for i, val in enumerate(mean_accuracies_plot):
            plt.text(i, val + std_accuracies_plot[i] + 0.02, f"{val:.3f}", ha='center', va='bottom')
        plt.savefig(args.output_plot_path)
        print(f"Saved vanilla test plot to {args.output_plot_path}")

    print("\nVanilla model testing and aggregation finished.")

if __name__ == "__main__":
    main()
