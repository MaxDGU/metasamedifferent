#!/usr/bin/env python
# naturalistic/plot_comparative_results.py

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Generate a comparative bar chart of MAML vs Vanilla SGD test accuracies.")
    parser.add_argument('--maml_json_path', type=str, required=True, help='Path to the MAML test results JSON file (e.g., /path/to/test_results_meta/meta_test_summary.json).')
    parser.add_argument('--vanilla_json_path', type=str, required=True, help='Path to the Vanilla SGD test results JSON file (e.g., /path/to/test_results_vanilla/vanilla_test_summary.json).')
    parser.add_argument('--output_plot_path', type=str, required=True, help='Path to save the comparative plot (e.g., comparison_plot.png).')
    parser.add_argument('--title', type=str, default='MAML vs. Vanilla SGD: Mean Test Accuracy', help='Title for the plot.')
    return parser.parse_args()

def load_results(json_path):
    """Loads results from a JSON file."""
    try:
        with open(json_path, 'r') as f:
            results = json.load(f)
        print(f"Successfully loaded results from: {json_path}")
        return results
    except FileNotFoundError:
        print(f"ERROR: JSON file not found at {json_path}. Cannot generate plot.")
        return None
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from {json_path}. Check file integrity.")
        return None

def main():
    args = parse_args()

    maml_results = load_results(args.maml_json_path)
    vanilla_results = load_results(args.vanilla_json_path)

    if maml_results is None or vanilla_results is None:
        print("Aborting plot generation due to missing or invalid JSON files.")
        return

    # Assume architectures are consistent and present in both files
    # Use MAML results as the source for architecture names and order
    architectures = list(maml_results.keys())
    if not architectures:
        print("No architectures found in MAML results. Cannot generate plot.")
        return
    
    print(f"Architectures found: {architectures}")

    maml_means = []
    maml_stds = []
    vanilla_means = []
    vanilla_stds = []

    for arch in architectures:
        if arch not in vanilla_results:
            print(f"WARNING: Architecture '{arch}' found in MAML results but not in Vanilla results. It will be skipped for plotting.")
            continue
        
        maml_means.append(maml_results[arch].get('mean_accuracy', 0))
        maml_stds.append(maml_results[arch].get('std_accuracy', 0))
        vanilla_means.append(vanilla_results[arch].get('mean_accuracy', 0))
        vanilla_stds.append(vanilla_results[arch].get('std_accuracy', 0))

    if not maml_means: # if all architectures were skipped
        print("No common architectures with data found to plot.")
        return

    # Ensure the output directory for the plot exists
    Path(args.output_plot_path).parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(maml_means))  # the label locations (using only common architectures)
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 7)) # Adjusted figure size for better readability
    rects1 = ax.bar(x - width/2, maml_means, width, label='MAML', yerr=maml_stds, capsize=5, color='#1f77b4') # Matplotlib default blue
    rects2 = ax.bar(x + width/2, vanilla_means, width, label='Vanilla SGD', yerr=vanilla_stds, capsize=5, color='#ff7f0e') # Matplotlib default orange

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Mean Test Accuracy')
    ax.set_xlabel('Model Architecture')
    ax.set_title(args.title)
    ax.set_xticks(x)
    # Use the actual architecture names that were plotted
    plotted_architectures = [arch for arch in architectures if arch in vanilla_results] 
    ax.set_xticklabels(plotted_architectures)
    ax.legend()
    ax.set_ylim(0, 1.05) # Set y-axis limit, slightly above 1.0 for text

    def autolabel(rects, std_devs):
        """Attach a text label above each bar in *rects*, displaying its height and std dev."""
        for i, rect in enumerate(rects):
            height = rect.get_height()
            ax.annotate(f'{height:.3f}\n(±{std_devs[i]:.3f})', # Display mean and std dev
                        xy=(rect.get_x() + rect.get_width() / 2, height + std_devs[i] + 0.01), # Position above error bar
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    autolabel(rects1, maml_stds)
    autolabel(rects2, vanilla_stds)

    fig.tight_layout() # Adjust layout to make room for labels
    plt.savefig(args.output_plot_path)
    print(f"Comparative plot saved to: {args.output_plot_path}")
    # plt.show() # Uncomment to display plot if running interactively

if __name__ == '__main__':
    main() 