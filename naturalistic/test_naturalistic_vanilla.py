import os
import torch
import json
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import gc
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import sys
import torch.nn as nn
import torch.nn.functional as F


# Import models from their respective files
from conv2 import SameDifferentCNN as VanillaConv2CNN
from conv4 import SameDifferentCNN as VanillaConv4CNN
from conv6 import SameDifferentCNN as VanillaConv6CNN

class NaturalisticDataset(Dataset):
    def __init__(self, data_dir, split='test'):
        self.data_dir = data_dir
        self.split = split
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
        
        # Load all image paths and labels
        self.same_dir = os.path.join(data_dir, split, 'same')
        self.diff_dir = os.path.join(data_dir, split, 'different')
        
        self.same_images = [os.path.join(self.same_dir, f) for f in os.listdir(self.same_dir) if f.endswith('.png')]
        self.diff_images = [os.path.join(self.diff_dir, f) for f in os.listdir(self.diff_dir) if f.endswith('.png')]
        
        # Ensure equal number of same and different examples
        min_count = min(len(self.same_images), len(self.diff_images))
        self.same_images = self.same_images[:min_count]
        self.diff_images = self.diff_images[:min_count]
        
        self.image_paths = self.same_images + self.diff_images
        self.labels = [1] * len(self.same_images) + [0] * len(self.diff_images)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and preprocess image
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        return image, label

def test_model(model, test_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            
            try:
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Clear memory after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"WARNING: GPU OOM error. Skipping batch...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                    continue
                else:
                    raise e
    
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True,
                      help='Base directory containing trained model checkpoints')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing naturalistic dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save test results')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size for testing')
    args = parser.parse_args()
    
    try:
        # Check for CUDA and set memory management
        if not torch.cuda.is_available():
            print("WARNING: CUDA is not available. Running on CPU.")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
            # Set CUDA memory management settings
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        print(f"Using device: {device}")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Define architectures and seeds
        architectures = ['conv2', 'conv4', 'conv6']
        seeds = range(42, 52)  # seeds 42 to 51
        
        all_results = {}
        
        # Process each architecture and seed combination
        for arch in architectures:
            all_results[arch] = {}
            print(f"\nProcessing architecture: {arch}")
            
            for seed in seeds:
                print(f"\nProcessing seed: {seed}")
                
                try:
                    # Clear memory before loading new model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    # Create model and load checkpoint to CPU first
                    if arch == 'conv2':
                        model = VanillaConv2CNN()
                    elif arch == 'conv4':
                        model = VanillaConv4CNN()
                    else:  # conv6
                        model = VanillaConv6CNN()
                    
                    model_path = os.path.join(args.model_dir, 'all_tasks', arch, 'test_lines', f'seed_{seed}', 'best_model.pt')
                    checkpoint = torch.load(model_path, map_location='cpu')
                    model.load_state_dict(checkpoint['model_state_dict'])
                    
                    # Move model to GPU
                    model = model.to(device)
                    print(f"Loaded model from {model_path}")
                    
                    # Create test dataset and dataloader with memory-efficient settings
                    test_dataset = NaturalisticDataset(args.data_dir, 'test')
                    test_loader = DataLoader(
                        test_dataset, 
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=1,
                        pin_memory=True
                    )
                    
                    # Test model
                    test_loss, test_acc = test_model(model, test_loader, device)
                    
                    # Store results
                    all_results[arch][str(seed)] = {
                        'test_loss': test_loss,
                        'test_accuracy': test_acc
                    }
                    
                    print(f"Test Loss: {test_loss:.4f}")
                    print(f"Test Accuracy: {test_acc:.4f}")
                    
                    # Clear memory after testing
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                except Exception as e:
                    print(f"Error processing seed {seed}: {str(e)}")
                    continue
        
        # Save all results
        results_file = os.path.join(args.output_dir, 'test_results.json')
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=4)
        
        print(f"\nResults saved to {results_file}")
        
        # Print summary
        print("\nSummary of results:")
        for arch in architectures:
            print(f"\nArchitecture: {arch}")
            for seed in seeds:
                if str(seed) in all_results[arch]:
                    results = all_results[arch][str(seed)]
                    print(f"  Seed {seed}:")
                    print(f"    Test Loss: {results['test_loss']:.4f}")
                    print(f"    Test Accuracy: {results['test_accuracy']:.4f}")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise

if __name__ == '__main__':
    main() 