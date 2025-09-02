import os
import torch
import json
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import learn2learn as l2l
import torch.nn.functional as F
import gc
from baselines.models.conv6 import SameDifferentCNN
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class NaturalisticDataset(Dataset):
    """Dataset for naturalistic same/different classification."""
    
    def __init__(self, root_dir, split='test'):
        """
        Args:
            root_dir: Path to N_16 directory
            split: One of 'train', 'val', or 'test'
        """
        self.root_dir = Path(root_dir)
        self.split = split
        
        # Get all image paths and labels
        same_dir = self.root_dir / split / 'same'
        diff_dir = self.root_dir / split / 'different'
        
        same_files = list(same_dir.glob('*.png'))
        diff_files = list(diff_dir.glob('*.png'))
        
        self.file_paths = same_files + diff_files
        self.labels = ([1] * len(same_files)) + ([0] * len(diff_files))
        
        # Convert to tensor
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        
        # Define transforms - using simple normalization without ImageNet stats
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],  # Simple normalization
                               std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        return {
            'image': image,
            'label': label
        }

def test_model(model, test_loader, device):
    """Test model on naturalistic dataset with memory-efficient batching."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    # Track prediction statistics
    all_predictions = []
    all_labels = []
    all_logits = []
    
    criterion = torch.nn.CrossEntropyLoss()
    
    print(f"\nTesting on naturalistic dataset")
    print(f"Dataset size: {len(test_loader.dataset)} images")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing batches"):
            try:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                
                # Store predictions and logits
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_logits.extend(outputs.cpu().numpy())
                
                # Update metrics
                total_loss += loss.item() * images.size(0)
                total_correct += correct
                total_samples += images.size(0)
                
                # Clear cache after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("WARNING: GPU OOM in batch, skipping...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    
    # Convert to numpy arrays for analysis
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)
    
    # Calculate prediction statistics
    pred_dist = np.bincount(all_predictions)
    label_dist = np.bincount(all_labels)
    
    # Calculate confidence (using softmax probabilities)
    probs = F.softmax(torch.tensor(all_logits), dim=1)
    confidences = torch.max(probs, dim=1)[0].numpy()
    avg_confidence = np.mean(confidences)
    
    print("\nDetailed Statistics:")
    print(f"Prediction distribution: {pred_dist}")
    print(f"Label distribution: {label_dist}")
    print(f"Average prediction confidence: {avg_confidence:.4f}")
    print(f"Confidence distribution:")
    print(f"  < 0.5: {np.mean(confidences < 0.5):.2%}")
    print(f"  0.5-0.7: {np.mean((confidences >= 0.5) & (confidences < 0.7)):.2%}")
    print(f"  0.7-0.9: {np.mean((confidences >= 0.7) & (confidences < 0.9)):.2%}")
    print(f"  >= 0.9: {np.mean(confidences >= 0.9):.2%}")
    
    return avg_loss, avg_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True,
                      help='Directory containing trained model checkpoints')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing naturalistic dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save test results')
    parser.add_argument('--batch_size', type=int, default=8,  # Reduced default batch size
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
        
        # Find all seed directories
        seed_dirs = [d for d in os.listdir(args.model_dir) 
                    if os.path.isdir(os.path.join(args.model_dir, d)) and d.startswith('seed_')]
        
        if not seed_dirs:
            raise ValueError(f"No seed directories found in {args.model_dir}")
        
        print(f"Found {len(seed_dirs)} seed directories")
        
        all_results = {}
        
        # Process each seed
        for seed_dir in seed_dirs:
            seed = seed_dir.replace('seed_', '')
            print(f"\nProcessing seed: {seed}")
            
            try:
                # Clear memory before loading new model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # Create model and load checkpoint to CPU first
                model = SameDifferentCNN()
                model_path = os.path.join(args.model_dir, seed_dir, 'best_model.pt')
                print(f"Attempting to load model from: {model_path}")
                
                # Load and print checkpoint structure
                checkpoint = torch.load(model_path, map_location='cpu')
                print("Checkpoint keys:", checkpoint.keys() if isinstance(checkpoint, dict) else "Not a dict")
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                # Move model to GPU
                model = model.to(device)
                print(f"Successfully loaded model from {model_path}")
                
                # Print model structure to verify
                print("\nModel structure:")
                for name, param in model.named_parameters():
                    print(f"{name}: {param.shape}")
                
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
                all_results[seed] = {
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
        for seed, results in all_results.items():
            print(f"Seed {seed}:")
            print(f"  Test Loss: {results['test_loss']:.4f}")
            print(f"  Test Accuracy: {results['test_accuracy']:.4f}")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise

if __name__ == '__main__':
    main() 