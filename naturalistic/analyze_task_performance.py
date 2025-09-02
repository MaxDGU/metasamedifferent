import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import re
from collections import defaultdict
from torchvision import transforms
import random
from baselines.models.conv2 import SameDifferentCNN as Conv2Model, load_model as load_conv2
from baselines.models.conv4 import SameDifferentCNN as Conv4Model, load_model as load_conv4
from baselines.models.conv6 import SameDifferentCNN as Conv6Model, load_model as load_conv6

class NaturalisticDataset(Dataset):
    def __init__(self, root_dir, transform=None, subset_fraction=0.25):
        self.root_dir = root_dir
        self.transform = transform
        
        # Get all image files
        self.image_files = []
        self.tasks = defaultdict(list)
        
        # Walk through the directory structure
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.png'):
                    full_path = os.path.join(root, file)
                    self.image_files.append(full_path)
                    
                    # Extract task identifier (e.g., '8024597_mouse-a' from '8024597_mouse-a_0.png')
                    task_id = re.match(r'(\d+_[a-zA-Z0-9-]+)_\d+\.png', file)
                    if task_id:
                        self.tasks[task_id.group(1)].append(full_path)
        
        # Randomly select a subset of tasks
        all_tasks = list(self.tasks.keys())
        num_tasks = len(all_tasks)
        num_subset_tasks = int(num_tasks * subset_fraction)
        selected_tasks = random.sample(all_tasks, num_subset_tasks)
        
        # Keep only images from selected tasks
        self.image_files = []
        for task in selected_tasks:
            self.image_files.extend(self.tasks[task])
        
        print(f"Selected {num_subset_tasks} tasks out of {num_tasks} total tasks")
        print(f"Total images in subset: {len(self.image_files)}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Extract label from path ('same' or 'different')
        label = 1 if 'same' in img_path else 0
        
        return image, label, img_path  # Added img_path to return value

def evaluate_model_on_tasks(model, dataloader, device):
    model.eval()
    task_performance = defaultdict(list)
    task_examples = defaultdict(list)  # Store example paths without affecting evaluation
    
    with torch.no_grad():
        for images, labels, img_paths in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Group predictions by task
            for i, (image_path, pred, label) in enumerate(zip(dataloader.dataset.image_files, predicted, labels)):
                task_id = re.match(r'(\d+_[a-zA-Z0-9-]+)_\d+\.png', os.path.basename(image_path))
                if task_id:
                    task_id = task_id.group(1)
                    task_performance[task_id].append((pred.item(), label.item()))
            
            # Store example images for visualization without affecting evaluation
            for img_path, pred, label in zip(img_paths, predicted, labels):
                task_id = re.match(r'(\d+_[a-zA-Z0-9-]+)_\d+\.png', os.path.basename(img_path))
                if task_id:
                    task_id = task_id.group(1)
                    task_examples[task_id].append((img_path, pred.item(), label.item()))
    
    # Calculate accuracy for each task
    task_accuracies = {}
    for task_id, results in task_performance.items():
        correct = sum(1 for pred, label in results if pred == label)
        total = len(results)
        task_accuracies[task_id] = correct / total
    
    return task_accuracies, task_examples

def display_task_examples(task_accuracies, task_predictions, model_name, dataset):
    # Get best and worst performing tasks
    sorted_tasks = sorted(task_accuracies.items(), key=lambda x: x[1])
    worst_task = sorted_tasks[0][0]
    best_task = sorted_tasks[-1][0]
    
    # Get example images for each task
    worst_examples = task_predictions[worst_task]
    best_examples = task_predictions[best_task]
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f'{model_name} Best and Worst Performing Tasks', fontsize=16)
    
    # Display worst task examples
    worst_same = next((img for img, pred, label in worst_examples if 'same' in img and label == 1), None)
    worst_diff = next((img for img, pred, label in worst_examples if 'different' in img and label == 0), None)
    
    if worst_same:
        img = Image.open(worst_same).convert('RGB')
        axes[0, 0].imshow(img)
        axes[0, 0].set_title(f'Worst Task ({worst_task})\nSame Example')
        axes[0, 0].axis('off')
    
    if worst_diff:
        img = Image.open(worst_diff).convert('RGB')
        axes[0, 1].imshow(img)
        axes[0, 1].set_title(f'Worst Task ({worst_task})\nDifferent Example')
        axes[0, 1].axis('off')
    
    # Display best task examples
    best_same = next((img for img, pred, label in best_examples if 'same' in img and label == 1), None)
    best_diff = next((img for img, pred, label in best_examples if 'different' in img and label == 0), None)
    
    if best_same:
        img = Image.open(best_same).convert('RGB')
        axes[1, 0].imshow(img)
        axes[1, 0].set_title(f'Best Task ({best_task})\nSame Example')
        axes[1, 0].axis('off')
    
    if best_diff:
        img = Image.open(best_diff).convert('RGB')
        axes[1, 1].imshow(img)
        axes[1, 1].set_title(f'Best Task ({best_task})\nDifferent Example')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'task_examples_{model_name.lower()}.png', dpi=300)
    plt.close()

def plot_task_performance(task_accuracies, model_name):
    # Sort tasks by accuracy
    sorted_tasks = sorted(task_accuracies.items(), key=lambda x: x[1])
    tasks, accuracies = zip(*sorted_tasks)
    
    # Create bar plot
    plt.figure(figsize=(15, 6))
    bars = plt.bar(range(len(tasks)), accuracies)
    
    # Add labels and title
    plt.title(f'{model_name} Performance by Task', fontsize=14)
    plt.xlabel('Task ID', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    
    # Add chance level line
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Chance level')
    
    # Rotate x-axis labels for better readability
    plt.xticks(range(len(tasks)), tasks, rotation=45, ha='right')
    
    # Add value labels on top of each bar
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # Add legend
    plt.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'task_performance_{model_name.lower()}.png', dpi=300)
    plt.close()
    
    # Print summary statistics
    print(f"\n{model_name} Task Performance Summary:")
    print(f"Mean Accuracy: {np.mean(accuracies):.4f}")
    print(f"Std Accuracy: {np.std(accuracies):.4f}")
    print(f"Min Accuracy: {min(accuracies):.4f}")
    print(f"Max Accuracy: {max(accuracies):.4f}")
    
    # Print tasks with particularly high or low performance
    print(f"\n{model_name} Top 5 Performing Tasks:")
    for task, acc in sorted(task_accuracies.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"{task}: {acc:.4f}")
    
    print(f"\n{model_name} Bottom 5 Performing Tasks:")
    for task, acc in sorted(task_accuracies.items(), key=lambda x: x[1])[:5]:
        print(f"{task}: {acc:.4f}")

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Define data transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create dataset and dataloader
    dataset = NaturalisticDataset(
        root_dir='naturalistic/objectsall_2/aligned/N_16/trainsize_6400_1200-300-100/test',
        transform=transform,
        subset_fraction=0.25
    )
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define models and their weight files
    models = {
        'Conv2': (Conv2Model(), 'meta_baseline/model_seed_0_pretesting_conv2.pt', load_conv2),
        'Conv4': (Conv4Model(), 'meta_baseline/model_seed_0_pretesting_conv4.pt', load_conv4),
        'Conv6': (Conv6Model(), 'weight_space_analysis/best_model_meta6.pt', load_conv6)
    }
    
    # Evaluate each model
    for model_name, (model, weight_file, load_func) in models.items():
        print(f"\nLoading {model_name} model...")
        model = load_func(model, weight_file, device)
        
        print(f"Evaluating {model_name} on tasks...")
        task_accuracies, task_examples = evaluate_model_on_tasks(model, dataloader, device)
        
        print(f"Plotting results for {model_name}...")
        plot_task_performance(task_accuracies, model_name)
        
        print(f"Displaying example images for {model_name}...")
        display_task_examples(task_accuracies, task_examples, model_name, dataset)
        
        # Save results to JSON
        results_file = f'{model_name.lower()}_task_results.json'
        with open(results_file, 'w') as f:
            json.dump(task_accuracies, f, indent=4)
        print(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()