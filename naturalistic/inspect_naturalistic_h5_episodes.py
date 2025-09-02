import h5py
import numpy as np
from pathlib import Path
from collections import Counter
import argparse
import matplotlib.pyplot as plt

def analyze_h5_file(h5_path, visualize_one_episode=False):
    """
    Analyzes a single HDF5 file containing meta-learning episodes.

    Args:
        h5_path (Path): Path to the HDF5 file.
        visualize_one_episode (bool): If True, captures data for one episode for visualization.

    Returns:
        dict: A dictionary containing analysis results for the file.
              Returns None if the file cannot be processed.
    """
    if not h5_path.exists():
        print(f"Error: File not found: {h5_path}")
        return None

    results = {
        "file_name": h5_path.name,
        "total_episodes": 0,
        "support_set_sizes": Counter(),
        "query_set_sizes": Counter(),
        "support_labels_all": [],
        "query_labels_all": [],
        "support_label_patterns": Counter(),
        "query_label_patterns": Counter(),
        "support_label_positions": {},
        "query_label_positions": {},
        "visualization_data": None
    }

    common_support_size = None
    common_query_size = None

    try:
        with h5py.File(h5_path, 'r') as f:
            episode_keys = sorted([k for k in f.keys() if k.startswith('episode_')])
            results["total_episodes"] = len(episode_keys)

            if not episode_keys:
                print(f"No episodes found in {h5_path.name}")
                return results

            temp_support_sizes = Counter()
            temp_query_sizes = Counter()
            for episode_key_idx, episode_key in enumerate(episode_keys):
                ep_group = f[episode_key]
                s_labels_raw = ep_group['support_labels'][()]
                q_labels_raw = ep_group['query_labels'][()]

                temp_support_sizes[len(s_labels_raw)] += 1
                temp_query_sizes[len(q_labels_raw)] += 1

                if visualize_one_episode and episode_key_idx == 0: # Only visualize the first episode encountered
                    try:
                        s_images_all_ep = ep_group['support_images'][()]
                        q_images_all_ep = ep_group['query_images'][()]
                        s_labels_all_ep = ep_group['support_labels'][()].astype(int)
                        q_labels_all_ep = ep_group['query_labels'][()].astype(int)
                        
                        if s_images_all_ep.shape[0] > 0 and q_images_all_ep.shape[0] > 0:
                            results["visualization_data"] = {
                                'support_images': s_images_all_ep,
                                'support_labels': s_labels_all_ep,
                                'query_images': q_images_all_ep,
                                'query_labels': q_labels_all_ep,
                                'episode_key': episode_key
                            }
                        visualize_one_episode = False # Ensure we only capture one episode for viz per file call
                    except KeyError as e:
                        print(f"Warning: Could not load full episode for visualization from {episode_key} in {h5_path.name}: {e}")
                    except IndexError as e:
                         print(f"Warning: Not enough images/labels in first episode of {h5_path.name} for visualization: {e}")

            if temp_support_sizes:
                common_support_size = temp_support_sizes.most_common(1)[0][0]
                print(f"  Most common support set size for {h5_path.name}: {common_support_size} (details: {temp_support_sizes})")
            if temp_query_sizes:
                common_query_size = temp_query_sizes.most_common(1)[0][0]
                print(f"  Most common query set size for {h5_path.name}: {common_query_size} (details: {temp_query_sizes})")

            for episode_key in episode_keys:
                ep_group = f[episode_key]
                support_labels = ep_group['support_labels'][()].astype(int)
                query_labels = ep_group['query_labels'][()].astype(int)
                s_size = len(support_labels)
                q_size = len(query_labels)
                results["support_set_sizes"][s_size] += 1
                results["query_set_sizes"][q_size] += 1
                results["support_labels_all"].extend(support_labels.tolist())
                results["query_labels_all"].extend(query_labels.tolist())
                if s_size == common_support_size:
                    results["support_label_patterns"][tuple(support_labels)] += 1
                    if common_support_size not in results["support_label_positions"]:
                        results["support_label_positions"][common_support_size] = [Counter() for _ in range(common_support_size)]
                    for i, label in enumerate(support_labels):
                        results["support_label_positions"][common_support_size][i][label] += 1
                if q_size == common_query_size:
                    results["query_label_patterns"][tuple(query_labels)] += 1
                    if common_query_size not in results["query_label_positions"]:
                        results["query_label_positions"][common_query_size] = [Counter() for _ in range(common_query_size)]
                    for i, label in enumerate(query_labels):
                        results["query_label_positions"][common_query_size][i][label] += 1
        print(f"Successfully processed {h5_path.name}")
    except Exception as e:
        print(f"Error processing file {h5_path.name}: {e}")
        return None
    return results

def print_analysis_results(results, visualize_one_episode_flag=False): # Renamed flag for clarity
    if not results:
        return

    print(f"\n--- Analysis for: {results['file_name']} ---")
    print(f"Total Episodes: {results['total_episodes']}")

    print("\nSupport Set Sizes:")
    for size, count in sorted(results["support_set_sizes"].items()):
        print(f"  Size {size}: {count} episodes")

    print("\nQuery Set Sizes:")
    for size, count in sorted(results["query_set_sizes"].items()):
        print(f"  Size {size}: {count} episodes")

    if results["support_labels_all"]:
        s_label_counts = Counter(results["support_labels_all"])
        s_total_labels = len(results["support_labels_all"])
        print("\nSupport Set Label Distribution (Overall):")
        for label, count in s_label_counts.items():
            print(f"  Label {label}: {count} ({count/s_total_labels*100:.2f}%)")
    
    if results["query_labels_all"]:
        q_label_counts = Counter(results["query_labels_all"])
        q_total_labels = len(results["query_labels_all"])
        print("\nQuery Set Label Distribution (Overall):")
        for label, count in q_label_counts.items():
            print(f"  Label {label}: {count} ({count/q_total_labels*100:.2f}%)")

    if results["support_label_patterns"]:
        common_s_size = next(iter(results["support_label_positions"]), None)
        if common_s_size is not None:
            print(f"\nSupport Set Label Patterns (for most common size: {common_s_size}):")
            total_common_s_patterns = sum(results["support_label_patterns"].values())
            if total_common_s_patterns > 0:
                for pattern, count in sorted(results["support_label_patterns"].items(), key=lambda item: item[1], reverse=True)[:10]:
                    print(f"  Pattern {pattern}: {count} ({count/total_common_s_patterns*100:.2f}%)")
                if len(results["support_label_patterns"]) > 10:
                    print("  ... (and more)")
            print(f"\nSupport Set Label Positional Bias (for most common size: {common_s_size}):")
            for i, pos_counter in enumerate(results["support_label_positions"][common_s_size]):
                total_at_pos = sum(pos_counter.values())
                if total_at_pos > 0:
                    print(f"  Position {i}:")
                    for label, count in sorted(pos_counter.items()):
                        print(f"    Label {label}: {count} ({count/total_at_pos*100:.2f}%)")
    
    if results["query_label_patterns"]:
        common_q_size = next(iter(results["query_label_positions"]), None)
        if common_q_size is not None:
            print(f"\nQuery Set Label Patterns (for most common size: {common_q_size}):")
            total_common_q_patterns = sum(results["query_label_patterns"].values())
            if total_common_q_patterns > 0:
                for pattern, count in sorted(results["query_label_patterns"].items(), key=lambda item: item[1], reverse=True)[:10]:
                    print(f"  Pattern {pattern}: {count} ({count/total_common_q_patterns*100:.2f}%)")
                if len(results["query_label_patterns"]) > 10:
                    print("  ... (and more)")
            print(f"\nQuery Set Label Positional Bias (for most common size: {common_q_size}):")
            for i, pos_counter in enumerate(results["query_label_positions"][common_q_size]):
                total_at_pos = sum(pos_counter.values())
                if total_at_pos > 0:
                    print(f"  Position {i}:")
                    for label, count in sorted(pos_counter.items()):
                        print(f"    Label {label}: {count} ({count/total_at_pos*100:.2f}%)")

    if visualize_one_episode_flag and results["visualization_data"]:
        vis_data = results["visualization_data"]
        s_images = vis_data['support_images']
        s_labels = vis_data['support_labels']
        q_images = vis_data['query_images']
        q_labels = vis_data['query_labels']
        
        num_s_images = s_images.shape[0]
        num_q_images = q_images.shape[0]
        total_images_to_plot = num_s_images + num_q_images

        if total_images_to_plot == 0:
            print("No images to visualize for the first episode.")
            return

        # Determine grid size (aim for a somewhat balanced layout, max 5 columns)
        ncols = min(5, total_images_to_plot) 
        nrows = (total_images_to_plot + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
        fig.suptitle(f"Full First Episode ({vis_data['episode_key']}) from {results['file_name']}", fontsize=16)
        axes = axes.flatten() # Flatten to 1D array for easy indexing

        img_idx = 0
        # Plot Support Images
        for i in range(num_s_images):
            axes[img_idx].imshow(s_images[i])
            axes[img_idx].set_title(f"Support {i}\nLabel: {s_labels[i]}", fontsize=10)
            axes[img_idx].axis('off')
            img_idx += 1
        
        # Plot Query Images
        for i in range(num_q_images):
            axes[img_idx].imshow(q_images[i])
            axes[img_idx].set_title(f"Query {i}\nLabel: {q_labels[i]}", fontsize=10)
            axes[img_idx].axis('off')
            img_idx += 1
        
        # Turn off any remaining empty subplots
        for i in range(img_idx, len(axes)):
            axes[i].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Analyze Naturalistic HDF5 episode structure and labels.")
    parser.add_argument('--data_dir', type=str, default='data/naturalistic',
                        help='Directory containing train.h5, val.h5, test.h5')
    parser.add_argument('--visualize_episode', action='store_true',
                        help='If set, visualizes all support/query images from the first episode of each HDF5 file.')
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    files_to_analyze = ["train.h5", "val.h5", "test.h5"]

    for file_name in files_to_analyze:
        h5_file_path = data_path / file_name
        analysis_data = analyze_h5_file(h5_file_path, visualize_one_episode=args.visualize_episode)
        if analysis_data:
            print_analysis_results(analysis_data, visualize_one_episode_flag=args.visualize_episode)
            print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main() 