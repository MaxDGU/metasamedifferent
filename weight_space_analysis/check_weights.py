import torch
import os

def print_model_params(model_path):
    print(f"\nChecking weights from: {model_path}")
    # Load to CPU if CUDA is not available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    print("\nLayer shapes:")
    for name, param in state_dict.items():
        print(f"{name}: {param.shape}")
    
    return state_dict

# Load both weight files
irregular_dict = print_model_params("weight_space_analysis/best_model_lines.pt")
filled_dict = print_model_params("weight_space_analysis/best_model_arrows.pt")

# Compare the weights
print("\nComparing weights between files:")
for name in irregular_dict.keys():
    if name in filled_dict:
        # Check if shapes match
        if irregular_dict[name].shape != filled_dict[name].shape:
            print(f"Shape mismatch for {name}:")
            print(f"  Irregular: {irregular_dict[name].shape}")
            print(f"  Filled: {filled_dict[name].shape}")
        # Check if weights are different
        if not torch.allclose(irregular_dict[name], filled_dict[name], rtol=1e-5, atol=1e-5):
            print(f"Different weights for {name}")
            print(f"  Max difference: {torch.max(torch.abs(irregular_dict[name] - filled_dict[name]))}")
    else:
        print(f"Key {name} not found in filled model")