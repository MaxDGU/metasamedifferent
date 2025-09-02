import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compile_results(base_dir='results/naturalistic'):
    """Compile results from all architectures and seeds."""
    architectures = ['conv2', 'conv4', 'conv6']
    seeds = list(range(42, 47))  # 5 seeds
    
    all_results = []
    
    for arch in architectures:
        arch_results = []
        for seed in seeds:
            results_file = os.path.join(base_dir, arch, f'seed_{seed}', 'metrics.json')
            
            if not os.path.exists(results_file):
                print(f"Warning: No results found for {arch} seed {seed}")
                continue
                
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    
                result = {
                    'architecture': arch,
                    'seed': seed,
                    'test_accuracy': data['test_metrics']['accuracy'],
                    'test_loss': data['test_metrics']['loss'],
                    'best_val_accuracy': data['best_val_metrics']['accuracy'],
                    'best_val_loss': data['best_val_metrics']['loss'],
                    'epochs_trained': data['total_epochs']
                }
                arch_results.append(result)
                
            except Exception as e:
                print(f"Error processing {arch} seed {seed}: {str(e)}")
        
        if arch_results:
            all_results.extend(arch_results)
    
    return pd.DataFrame(all_results)

def plot_results(df, output_dir='results/naturalistic'):
    """Create visualizations of the results."""
    # Set style
    plt.style.use('seaborn')
    
    # 1. Box plot of test accuracy by architecture
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='architecture', y='test_accuracy', data=df)
    sns.swarmplot(x='architecture', y='test_accuracy', data=df, color='black', alpha=0.5)
    plt.title('Test Accuracy by Architecture')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Architecture')
    plt.savefig(os.path.join(output_dir, 'test_accuracy_by_arch.png'))
    plt.close()
    
    # 2. Bar plot with error bars
    summary = df.groupby('architecture').agg({
        'test_accuracy': ['mean', 'std'],
        'test_loss': ['mean', 'std']
    }).reset_index()
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(summary['architecture'], 
                  summary[('test_accuracy', 'mean')],
                  yerr=summary[('test_accuracy', 'std')],
                  capsize=5)
    plt.title('Mean Test Accuracy by Architecture')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Architecture')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.savefig(os.path.join(output_dir, 'mean_test_accuracy.png'))
    plt.close()
    
    return summary

def main():
    # Compile results
    results_df = compile_results()
    
    if results_df is None or len(results_df) == 0:
        print("No results found to analyze")
        return
    
    # Create summary
    summary = plot_results(results_df)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("==================")
    for _, row in summary.iterrows():
        arch = row['architecture']
        mean_acc = row[('test_accuracy', 'mean')]
        std_acc = row[('test_accuracy', 'std')]
        print(f"\n{arch}:")
        print(f"Test Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
    
    # Save summary to file
    summary_file = os.path.join('results/naturalistic', 'summary.csv')
    summary.to_csv(summary_file)
    print(f"\nSaved summary to {summary_file}")
    
    # Save full results
    full_results_file = os.path.join('results/naturalistic', 'all_results.csv')
    results_df.to_csv(full_results_file, index=False)
    print(f"Saved full results to {full_results_file}")

if __name__ == '__main__':
    main() 