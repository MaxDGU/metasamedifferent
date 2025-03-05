import os
import json
import numpy as np
from pathlib import Path
from models.utils import PB_TASKS, ARCHITECTURES, SEEDS


#this file TO BE RUN AFTER all seeds have been trained


def compile_results(base_dir='results/pb_baselines'):
    """Compile test accuracies across all architectures, tasks, and seeds."""
    results = {
        'by_architecture': {},
        'by_task': {},
        'overall': {
            'mean': None,
            'std': None,
            'n_total': 0
        }
    }
    
    all_accuracies = []
    
    for arch in ARCHITECTURES:
        results['by_architecture'][arch] = {
            'mean': None,
            'std': None,
            'n_total': 0,
            'by_task': {}
        }
        
        for task in PB_TASKS:
            results['by_architecture'][arch]['by_task'][task] = {
                'accuracies': [],
                'mean': None,
                'std': None,
                'n_seeds': 0
            }
    
    # Initialize task results
    for task in PB_TASKS:
        results['by_task'][task] = {
            'mean': None,
            'std': None,
            'n_total': 0,
            'by_architecture': {}
        }
        
        # Initialize architecture results for this task
        for arch in ARCHITECTURES:
            results['by_task'][task]['by_architecture'][arch] = {
                'accuracies': [],
                'mean': None,
                'std': None,
                'n_seeds': 0
            }
    
    # Collect results
    for arch in ARCHITECTURES:
        for task in PB_TASKS:
            for seed in SEEDS:
                metrics_path = os.path.join(
                    base_dir, 'all_tasks', arch,
                    f'test_{task}', f'seed_{seed}',
                    'metrics.json'
                )
                
                if os.path.exists(metrics_path):
                    try:
                        with open(metrics_path, 'r') as f:
                            metrics = json.load(f)
                            test_acc = metrics['test_acc']
                            
                            # Store accuracy in both views
                            results['by_architecture'][arch]['by_task'][task]['accuracies'].append(test_acc)
                            results['by_task'][task]['by_architecture'][arch]['accuracies'].append(test_acc)
                            all_accuracies.append(test_acc)
                            
                    except Exception as e:
                        print(f"Error reading {metrics_path}: {str(e)}")
    
    # Calculate statistics
    # 1. By architecture and task
    for arch in ARCHITECTURES:
        arch_accs = []
        for task in PB_TASKS:
            accs = results['by_architecture'][arch]['by_task'][task]['accuracies']
            if accs:
                mean = np.mean(accs)
                std = np.std(accs)
                n_seeds = len(accs)
                
                # Update architecture-task view
                results['by_architecture'][arch]['by_task'][task].update({
                    'mean': float(mean),
                    'std': float(std),
                    'n_seeds': n_seeds
                })
                
                # Update task-architecture view
                results['by_task'][task]['by_architecture'][arch].update({
                    'mean': float(mean),
                    'std': float(std),
                    'n_seeds': n_seeds
                })
                
                arch_accs.extend(accs)
        
        # Calculate architecture-level statistics
        if arch_accs:
            results['by_architecture'][arch].update({
                'mean': float(np.mean(arch_accs)),
                'std': float(np.std(arch_accs)),
                'n_total': len(arch_accs)
            })
    
    # 2. By task
    for task in PB_TASKS:
        task_accs = []
        for arch in ARCHITECTURES:
            task_accs.extend(results['by_task'][task]['by_architecture'][arch]['accuracies'])
        
        if task_accs:
            results['by_task'][task].update({
                'mean': float(np.mean(task_accs)),
                'std': float(np.std(task_accs)),
                'n_total': len(task_accs)
            })
    
    # 3. Overall statistics
    if all_accuracies:
        results['overall'].update({
            'mean': float(np.mean(all_accuracies)),
            'std': float(np.std(all_accuracies)),
            'n_total': len(all_accuracies)
        })
    
    return results

def main():
    # Compile results
    results = compile_results()
    
    # Save results
    output_dir = os.path.join('results/pb_baselines', 'compiled_results')
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'test_accuracies.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\nResults Summary:")
    print("=" * 50)
    
    print(f"\nOverall Statistics:")
    print(f"Mean accuracy: {results['overall']['mean']*100:.2f}% ± {results['overall']['std']*100:.2f}%")
    print(f"Total samples: {results['overall']['n_total']}")
    
    print("\nBy Architecture:")
    for arch in ARCHITECTURES:
        arch_results = results['by_architecture'][arch]
        print(f"\n{arch}:")
        print(f"Mean accuracy: {arch_results['mean']*100:.2f}% ± {arch_results['std']*100:.2f}%")
        print(f"Total samples: {arch_results['n_total']}")
        print("\nTask breakdown:")
        for task in PB_TASKS:
            task_results = arch_results['by_task'][task]
            if task_results['n_seeds'] > 0:
                print(f"  {task}: {task_results['mean']*100:.2f}% ± {task_results['std']*100:.2f}% (n={task_results['n_seeds']})")

if __name__ == '__main__':
    main() 