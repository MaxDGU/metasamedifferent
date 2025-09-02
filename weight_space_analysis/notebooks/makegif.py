import imageio.v2 as imageio
import glob
import os
from pathlib import Path

def create_layer_gif(image_dir, output_path, duration=1.0):
    """
    Create a GIF from layer visualization images.
    
    Args:
        image_dir (str): Directory containing the layer images
        output_path (str): Path where to save the GIF
        duration (float): Duration for each frame in seconds
    """
    # Get all PNG files and sort them by layer number
    images = []
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')), 
                        key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
    
    # Read images
    for filename in image_files:
        images.append(imageio.imread(filename))
    
    # Create GIF
    print(f"Creating GIF with {len(images)} frames...")
    imageio.mimsave(output_path, images, duration=duration)
    print(f"GIF saved to {output_path}")

# Example usage:
# If using the zip file contents:
create_layer_gif('weight_space_analysis/notebooks/tsne_init', 'layer_evolution.gif', duration=1.0)

# Or if using notebook outputs:
# create_layer_gif('path/to/notebook/output/directory', 'layer_evolution.gif', duration=1.0)