import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

def find_example_image(h5_filepath, target_label, specific_episode_key=None, specific_item_idx=None):
    """
    Finds an example query image with the target_label from the HDF5 file.

    Args:
        h5_filepath (Path): Path to the HDF5 file.
        target_label (int): 0 for 'different', 1 for 'same'.
        specific_episode_key (str, optional): Specific episode key to use (e.g., 'episode_000001').
        specific_item_idx (int, optional): Specific item index within the query set of the episode.

    Returns:
        np.ndarray: The image (H, W, C) as a NumPy array.
        str: The episode key from which the image was taken.
        int: The item index from which the image was taken.
    
    Raises:
        ValueError: If no suitable image is found.
    """
    with h5py.File(h5_filepath, 'r') as f:
        if specific_episode_key and specific_item_idx is not None:
            if specific_episode_key not in f:
                raise ValueError(f"Specified episode_key '{specific_episode_key}' not found in {h5_filepath}")
            episode_group = f[specific_episode_key]
            query_labels = episode_group['query_labels'][:]
            query_images = episode_group['query_images'][:]
            
            if 0 <= specific_item_idx < len(query_labels):
                if query_labels[specific_item_idx] == target_label:
                    print(f"Found specified example: Episode {specific_episode_key}, Item {specific_item_idx}, Label {target_label}")
                    return query_images[specific_item_idx], specific_episode_key, specific_item_idx
                else:
                    print(f"Warning: Specified item {specific_item_idx} in episode {specific_episode_key} has label {query_labels[specific_item_idx]}, not {target_label}. Searching randomly.")
            else:
                raise ValueError(f"Specified item_idx {specific_item_idx} out of range for episode {specific_episode_key} (len: {len(query_labels)})")

        episode_keys = sorted([k for k in f.keys() if k.startswith('episode_')])
        if not episode_keys:
            raise ValueError(f"No episodes found in {h5_filepath}")

        # Shuffle keys to get a random example if not specified
        random.shuffle(episode_keys)

        for episode_key in episode_keys:
            episode_group = f[episode_key]
            query_labels = episode_group['query_labels'][:]
            query_images = episode_group['query_images'][:]
            
            item_indices = list(range(len(query_labels)))
            random.shuffle(item_indices) # Shuffle indices within the episode

            for i in item_indices:
                if query_labels[i] == target_label:
                    print(f"Found random example: Episode {episode_key}, Item {i}, Label {target_label}")
                    return query_images[i], episode_key, i
        
    raise ValueError(f"Could not find an image with label {target_label} in {h5_filepath}")

def main():
    parser = argparse.ArgumentParser(description="Generate an illustration of 'Same' and 'Different' examples from a naturalistic dataset HDF5 file.")
    parser.add_argument('--h5_file', type=str, required=True,
                        help='Path to the HDF5 file (e.g., train.h5, val.h5).')
    parser.add_argument('--output_image', type=str, default='naturalistic_episode_illustration.png',
                        help='Path to save the output PNG image.')
    parser.add_argument('--same_episode_key', type=str, default=None, help="Optional: Specific episode key for the 'Same' example.")
    parser.add_argument('--same_item_idx', type=int, default=None, help="Optional: Specific item index for the 'Same' example.")
    parser.add_argument('--diff_episode_key', type=str, default=None, help="Optional: Specific episode key for the 'Different' example.")
    parser.add_argument('--diff_item_idx', type=int, default=None, help="Optional: Specific item index for the 'Different' example.")

    args = parser.parse_args()

    h5_filepath = Path(args.h5_file)
    if not h5_filepath.exists():
        print(f"Error: HDF5 file not found at {h5_filepath}")
        return

    try:
        # Find a "Same" example (label 1)
        same_image, same_ep_key, same_idx = find_example_image(h5_filepath, 1, args.same_episode_key, args.same_item_idx)
        
        # Find a "Different" example (label 0)
        # Try to ensure it's not the exact same image if randomly selected from same episode
        # For simplicity, we just search. If specific indices are given, it's up to the user.
        # If not specified, the random search should generally pick different ones.
        diff_image, diff_ep_key, diff_idx = find_example_image(h5_filepath, 0, args.diff_episode_key, args.diff_item_idx)

        # Check if by chance we got the same image for both if from same episode (unlikely with random search but possible if file is small)
        if same_ep_key == diff_ep_key and same_idx == diff_idx and not (args.same_episode_key and args.diff_episode_key):
             print("Warning: Randomly selected the same image for 'Same' and 'Different'. Attempting to find a different 'Different' example.")
             # Try one more time for different, explicitly avoiding the 'same' image's episode if it was chosen randomly
             # This is a simple retry, more robust logic might be needed for edge cases
             temp_diff_ep, temp_diff_idx = None, None
             all_episode_keys = []
             with h5py.File(h5_filepath, 'r') as f_retry:
                 all_episode_keys = sorted([k for k in f_retry.keys() if k.startswith('episode_')])
             
             # Try another episode if possible
             for retry_ep_key in all_episode_keys:
                 if retry_ep_key == same_ep_key and len(all_episode_keys) > 1: # If it's the same episode and there are others, skip
                     continue
                 try:
                     candidate_diff_image, candidate_diff_ep_key, candidate_diff_idx = find_example_image(h5_filepath, 0, retry_ep_key, None)
                     if not (candidate_diff_ep_key == same_ep_key and candidate_diff_idx == same_idx) :
                         diff_image, diff_ep_key, diff_idx = candidate_diff_image, candidate_diff_ep_key, candidate_diff_idx
                         break
                 except ValueError:
                     continue # Episode might not have a 'different' example

    except ValueError as e:
        print(f"Error: {e}")
        return

    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(8, 4)) # Poster: consider (10,5) or (12,6)

    # Display Same image
    axes[0].imshow(same_image, interpolation='bicubic')
    axes[0].set_title("Same", fontsize=20, pad=10)
    axes[0].axis('off')

    # Display Different image
    axes[1].imshow(diff_image, interpolation='bicubic')
    axes[1].set_title("Different", fontsize=20, pad=10)
    axes[1].axis('off')

    plt.tight_layout(pad=0.5) # Add some padding
    
    try:
        plt.savefig(args.output_image, dpi=300, bbox_inches='tight')
        print(f"Illustration saved to {args.output_image}")
        print(f"  'Same' example from: Episode {same_ep_key}, Item {same_idx}")
        print(f"  'Different' example from: Episode {diff_ep_key}, Item {diff_idx}")

    except Exception as e:
        print(f"Error saving image: {e}")

if __name__ == "__main__":
    main() 