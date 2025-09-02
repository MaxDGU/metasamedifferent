import torch
import sys
import os
from pathlib import Path

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baselines.models.conv2 import SameDifferentCNN as Conv2CNN
from baselines.models.conv4 import SameDifferentCNN as Conv4CNN

def inspect_checkpoint(checkpoint_path, model_type):
    print(f"\nInspecting {model_type} checkpoint at {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Print checkpoint structure
    print("\nCheckpoint structure:")
    if isinstance(checkpoint, dict):
        print("Keys in checkpoint:", checkpoint.keys())
        if 'model_state_dict' in checkpoint:
            print("\nKeys in model_state_dict:", checkpoint['model_state_dict'].keys())
        elif 'state_dict' in checkpoint:
            print("\nKeys in state_dict:", checkpoint['state_dict'].keys())
    else:
        print("Checkpoint is not a dictionary")
    
    # Create model and print its state dict keys
    if model_type == 'conv2':
        model = Conv2CNN()
    elif model_type == 'conv4':
        model = Conv4CNN()
    
    print("\nModel state dict keys:")
    print(model.state_dict().keys())
    
    # Try to load the weights and see what happens
    try:
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        print("\nSuccessfully loaded weights!")
    except Exception as e:
        print("\nError loading weights:", str(e))

def main():
    # Check conv2 weights
    conv2_path = 'meta_baseline/model_seed_0_pretesting_conv2.pt'
    if os.path.exists(conv2_path):
        inspect_checkpoint(conv2_path, 'conv2')
    
    # Check conv4 weights
    conv4_path = 'meta_baseline/model_seed_0_pretesting_conv4.pt'
    if os.path.exists(conv4_path):
        inspect_checkpoint(conv4_path, 'conv4')

if __name__ == "__main__":
    main() 