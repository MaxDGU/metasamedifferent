import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
plt.style.use('default')  # Use default style instead of seaborn
sns.set_theme()  # This will apply seaborn's styling

# Load results
with open('test_results_meta6.json', 'r') as f:
    meta_results = json.load(f)

with open('test_results_vanilla.json', 'r') as f:
    vanilla_results = json.load(f)

# Prepare data for plotting
seeds = sorted(list(meta_results.keys()))
architectures = ['conv2', 'conv4', 'conv6']

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Accuracy Comparison
# Meta-learned results
meta_accuracies = [meta_results[seed]['test_accuracy'] for seed in seeds]
ax1.plot(seeds, meta_accuracies, 'o-', label='Meta-learned Conv6', linewidth=2, markersize=8)

# Vanilla results
for arch in architectures:
    vanilla_accuracies = [vanilla_results[arch][seed]['test_accuracy'] for seed in seeds]
    ax1.plot(seeds, vanilla_accuracies, 'o-', label=f'Vanilla {arch}', linewidth=2, markersize=8)

ax1.set_xlabel('Seed')
ax1.set_ylabel('Test Accuracy')
ax1.set_title('Model Performance Comparison')
ax1.grid(True)
ax1.legend()
ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Chance Level')

# Plot 2: Loss Comparison
# Meta-learned results
meta_losses = [meta_results[seed]['test_loss'] for seed in seeds]
ax2.plot(seeds, meta_losses, 'o-', label='Meta-learned Conv6', linewidth=2, markersize=8)

# Vanilla results
for arch in architectures:
    vanilla_losses = [vanilla_results[arch][seed]['test_loss'] for seed in seeds]
    ax2.plot(seeds, vanilla_losses, 'o-', label=f'Vanilla {arch}', linewidth=2, markersize=8)

ax2.set_xlabel('Seed')
ax2.set_ylabel('Test Loss')
ax2.set_title('Loss Comparison')
ax2.grid(True)
ax2.legend()

# Adjust layout and save
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Print summary statistics
print("\nSummary Statistics:")
print("\nMeta-learned Conv6:")
print(f"Mean Accuracy: {np.mean(meta_accuracies):.4f}")
print(f"Std Accuracy: {np.std(meta_accuracies):.4f}")
print(f"Mean Loss: {np.mean(meta_losses):.4f}")
print(f"Std Loss: {np.std(meta_losses):.4f}")

print("\nVanilla Models:")
for arch in architectures:
    accuracies = [vanilla_results[arch][seed]['test_accuracy'] for seed in seeds]
    losses = [vanilla_results[arch][seed]['test_loss'] for seed in seeds]
    print(f"\n{arch}:")
    print(f"Mean Accuracy: {np.mean(accuracies):.4f}")
    print(f"Std Accuracy: {np.std(accuracies):.4f}")
    print(f"Mean Loss: {np.mean(losses):.4f}")
    print(f"Std Loss: {np.std(losses):.4f}") 