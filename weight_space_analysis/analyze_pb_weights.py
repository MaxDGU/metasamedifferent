import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
import sys
import collections # For defaultdict

# --- Add project root to sys.path ---
# Comment in script_dir and project_root was misleading.
# Assuming 'weight_space_analysis' is a top-level directory in the project.
# script_dir = Path(__file__).parent.resolve()
# project_root = script_dir.parent # This would be the project root
# sys.path.append(str(project_root)) # Prefer PYTHONPATH setup
# print(f"Added project root to sys.path: {project_root}")

# Import model definitions
# MODIFIED: Assuming models are in a sibling 'models' directory or installed
# Or adjust based on actual structure from where this script is run relative to models.
# Fallback to direct imports if they are in the same dir or PYTHONPATH is set up.
try:
    from baselines.models.conv2 import SameDifferentCNN as Conv2CNN
    from baselines.models.conv4 import SameDifferentCNN as Conv4CNN
    from baselines.models.conv6 import SameDifferentCNN as Conv6CNN
except ImportError:
    print("Warning: Could not import from baselines.models. Assuming models (conv2, conv4, conv6) are accessible directly or via PYTHONPATH.")
    # These direct imports might fail if the files aren't structured as such relative to this script.
    # This is a placeholder; actual model import strategy depends on the project structure.
    # If analyze_pb_weights.py is inside weight_space_analysis, and models are in sibling 'models' dir:
    # script_dir = Path(__file__).resolve().parent
    # project_root_guess = script_dir.parent
    # sys.path.insert(0, str(project_root_guess / 'models')) 
    # from conv2 import SameDifferentCNN as Conv2CNN # etc.
    # For now, assume they are somehow in path:
    from conv2 import SameDifferentCNN as Conv2CNN
    from conv4 import SameDifferentCNN as Conv4CNN
    from conv6 import SameDifferentCNN as Conv6CNN

# --- Constants (should match the run script) ---
ARCHITECTURES = {
    'conv2': Conv2CNN,
    'conv4': Conv4CNN,
    'conv6': Conv6CNN
}
SEEDS = list(range(10)) # Seeds 0 through 9
PB_TASKS = ['regular', 'lines', 'open', 'wider_line', 'scrambled',
            'random_color', 'arrows', 'irregular', 'filled', 'original']

# --- Paths for NEW MAML models (trained on all tasks, from newer experiments) ---
# Base subdirectory within args.maml_runs_base_dir where new MAML experiment folders are located.
NEW_MAML_RESULTS_SUBDIR = "maml_results_new_2" # MODIFIED: Aligning with user logs, was "maml_results_new_2ndorder"
# The actual experiment folders are like: exp_all_tasks_fomaml_conv6_seed8_Svar_Q3_20250527_210526
# And the target model file inside is 'best_model.pth'

# --- OLD MAML Path Definitions (Commented out as new structure is used) ---
# MAML_ALL_TASKS_PARENT_DIRS = {
#     'conv6': "exp1_(untested)conv6lr_runs_20250127_110352",
#     'conv4': "exp1_(finished)conv4lr_runs_20250126_201548",
#     'conv2': "conv2lr_runs_20250127_131933"
# }
# MAML_MODEL_FILENAME_PATTERNS = {
#     'conv6': "seed_{seed}/model_seed_{seed}.pt",
#     'conv4': "seed_{seed}/model_seed_{seed}.pt",
#     'conv2': "seed_{seed}/model_seed_{seed}_pretesting.pt"
# }

# Label for MAML model points
MAML_LABEL = "MAML (all tasks)"
# We will add vanilla model label later
# VANILLA_ALL_TASKS_LABEL = "Vanilla (all tasks)"

def flatten_weights(state_dict):
    """Flattens all parameters in a state_dict into a single numpy vector."""
    all_weights = []
    if state_dict is None: # ADDED: Handle None state_dict
        return None
    # Sort keys to ensure consistent order (optional but good practice)
    for key in sorted(state_dict.keys()):
        param = state_dict[key]
        if isinstance(param, torch.Tensor):
             # Ensure tensor is on CPU and converted to numpy
            all_weights.append(param.detach().cpu().numpy().flatten())
    if not all_weights:
        return None
    return np.concatenate(all_weights)

def print_state_dict_summary(state_dict, model_type_label): # ADDED_FUNCTION
    print(f"    DEBUG: State dict summary for {model_type_label}:")
    total_params = 0
    keys_found = 0
    if state_dict is None:
        print(f"    DEBUG: State dict for {model_type_label} is None.")
        return
    for k, v in sorted(state_dict.items()):
        keys_found +=1
        if isinstance(v, torch.Tensor):
            print(f"      Key: {k}, Shape: {v.shape}, Numel: {v.numel()}")
            total_params += v.numel()
        else:
            print(f"      Key: {k}, Type: {type(v)}")
    if keys_found == 0:
        print(f"    DEBUG: State dict for {model_type_label} is empty.")
    print(f"    DEBUG: Total parameters in {model_type_label} state_dict: {total_params}")

def main(args):
    base_results_dir = Path(args.results_dir)
    # New argument for the base directory where MAML multi-task models are.
    # This assumes MAML_ALL_TASKS_PARENT_DIRS contains paths *relative* to this maml_base_dir,
    # OR MAML_ALL_TASKS_PARENT_DIRS should store absolute paths.
    # Let's assume for now MAML_ALL_TASKS_PARENT_DIRS are full paths or easily discoverable.
    # We might need a new CLI arg if these MAML dirs are scattered.
    # For simplicity in this step, I'll assume paths in MAML_ALL_TASKS_PARENT_DIRS are resolvable as is.
    # If not, they need to be joined with a common base path provided by a new arg.
    # Let's add an argument for this.
    maml_runs_base_dir = Path(args.maml_runs_base_dir) if args.maml_runs_base_dir else None

    output_plot_dir = Path(args.output_plot_dir)
    
    output_plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"Analyzing weights across all PB tasks.")
    print(f"Loading results from: {base_results_dir}")
    if maml_runs_base_dir and NEW_MAML_RESULTS_SUBDIR:
        print(f"Loading NEW MAML models from: {maml_runs_base_dir / NEW_MAML_RESULTS_SUBDIR}")
    print(f"Saving plots to: {output_plot_dir}")

    # Use a colormap with enough distinct colors for tasks
    cmap = plt.get_cmap('tab10') 
    task_colors = {task: cmap(i) for i, task in enumerate(PB_TASKS)}
    # Add specific colors/markers for MAML and Vanilla (later)
    maml_plot_style = {'color': 'black', 'marker': '*', 's': 150, 'label': MAML_LABEL, 'alpha': 0.9, 'zorder': 10}
    # vanilla_plot_style = {'color': 'grey', 'marker': 's', 's': 100, 'label': VANILLA_ALL_TASKS_LABEL, 'alpha':0.9, 'zorder':9}

    # --- Process each architecture --- 
    for arch_name, model_class in ARCHITECTURES.items():
        print(f"\n===== Processing Architecture: {arch_name} ====")
        
        # Flags for printing state_dict summaries once per architecture
        printed_final_single_task_summary_for_current_arch = False
        printed_maml_summary_for_current_arch = False

        # Lists for FINAL single-task model weights (from best_model.pth)
        all_final_single_task_weight_vectors = []
        all_final_single_task_point_labels = [] # Stores task name
        all_final_single_task_seed_labels = []  # Store seed 
        all_final_single_task_point_categories = [] # Category: 'single_task_final'

        # Lists for INITIAL single-task model weights (from initial_model.pth)
        all_initial_single_task_weight_vectors = []
        all_initial_single_task_point_labels = []
        all_initial_single_task_seed_labels = []
        all_initial_single_task_point_categories = [] # Category: 'single_task_initial'

        # Lists for MAML (all tasks) model weights
        all_maml_weight_vectors = []
        all_maml_point_labels = []
        all_maml_seed_labels = []
        all_maml_point_categories = [] # Category: 'maml_all'
        
        found_final_weights = 0 # Renamed from found_weights
        found_initial_weights = 0
        maml_found_total_seeds = 0 # Initialize for MAML weights (will count new MAML)

        # --- Loop through tasks and seeds to collect SINGLE-TASK weights ---
        print(f"  Loading SINGLE-TASK (initial and final) weights for {arch_name}...")
        # Ensure PB_TASKS and SEEDS are defined and accessible here
        # PB_TASKS = ['regular', 'lines', ...] # Should be globally defined
        # SEEDS = list(range(10)) # Should be globally defined (original 0-9 indices)
        num_original_seeds = len(SEEDS)

        for task_idx, task_name in enumerate(PB_TASKS):
            task_found_final_seeds = 0
            task_found_initial_seeds = 0
            for original_seed_run_idx, original_seed_val in enumerate(SEEDS): # original_seed_val will be 0, 1, ..., 9
                # Calculate the globally unique seed that was used for folder naming
                globally_unique_seed = (task_idx * num_original_seeds) + original_seed_val

                # Construct paths using the globally_unique_seed for the folder name
                model_folder_path = base_results_dir / task_name / arch_name / f'seed_{globally_unique_seed}'
                final_model_path = model_folder_path / 'best_model.pth'
                initial_model_path = model_folder_path / 'initial_model.pth'

                # Load FINAL weights (best_model.pth)
                if final_model_path.exists():
                    try:
                        state_dict = torch.load(final_model_path, map_location=torch.device('cpu'), weights_only=True) # MODIFIED
                        
                        if arch_name == 'conv4' and not printed_final_single_task_summary_for_current_arch: # ADDED conditional print
                            print_state_dict_summary(state_dict, f"FINAL single-task {arch_name} ({task_name}, seed {original_seed_val})")
                            printed_final_single_task_summary_for_current_arch = True

                        flat_weights = flatten_weights(state_dict)
                        
                        if flat_weights is not None:
                            all_final_single_task_weight_vectors.append(flat_weights)
                            all_final_single_task_point_labels.append(task_name)
                            all_final_single_task_seed_labels.append(original_seed_val) # Store the original 0-9 seed index
                            all_final_single_task_point_categories.append('single_task_final')
                            found_final_weights += 1
                            task_found_final_seeds += 1
                    except Exception as e:
                        print(f"    Error loading FINAL single-task weights for {arch_name}, task {task_name}, seed_run_idx {original_seed_val} (folder seed_{globally_unique_seed}) from {final_model_path}: {e}")

                # Load INITIAL weights (initial_model.pth)
                if initial_model_path.exists():
                    try:
                        state_dict = torch.load(initial_model_path, map_location=torch.device('cpu'), weights_only=True) # MODIFIED
                        flat_weights = flatten_weights(state_dict)
                        
                        # DEBUG: Print info for the first few initial weights loaded per architecture
                        if task_found_initial_seeds < 2 and flat_weights is not None: 
                           print(f"    DEBUG (INITIAL single-task {arch_name}, task {task_name}, original_seed_run_idx {original_seed_val}, folder seed_{globally_unique_seed}):")
                           print(f"      Path: {initial_model_path}")
                           print(f"      flat_weights_norm: {np.linalg.norm(flat_weights)}")

                        if flat_weights is not None:
                            all_initial_single_task_weight_vectors.append(flat_weights)
                            all_initial_single_task_point_labels.append(task_name)
                            all_initial_single_task_seed_labels.append(original_seed_val) # Store the original 0-9 seed index
                            all_initial_single_task_point_categories.append('single_task_initial')
                            found_initial_weights += 1
                            task_found_initial_seeds += 1
                        else: # ADDED debug print
                            print(f"    Warning: flatten_weights returned None for INITIAL model at {initial_model_path} (state_dict was {'None' if state_dict is None else 'valid but resulted in no weights'})")
                    except Exception as e:
                        print(f"    Error loading INITIAL single-task weights for {arch_name}, task {task_name}, seed_run_idx {original_seed_val} (folder seed_{globally_unique_seed}) from {initial_model_path}: {e}")
                else: # ADDED debug print
                    print(f"    INITIAL model path NOT FOUND: {initial_model_path}")
            
        print(f"  Finished loading single-task weights for {arch_name}. Found {found_final_weights} FINAL vectors and {found_initial_weights} INITIAL vectors.")

        # --- Load NEW MAML (trained on all tasks) weights for the current architecture ---
        if args.maml_runs_base_dir and NEW_MAML_RESULTS_SUBDIR:
            print(f"  Loading NEW MAML (all tasks) weights for {arch_name} from subdir '{NEW_MAML_RESULTS_SUBDIR}'...")
            maml_runs_base_dir = Path(args.maml_runs_base_dir) # Ensure it's a Path object
            new_maml_arch_base_path = maml_runs_base_dir / NEW_MAML_RESULTS_SUBDIR
            
            if not new_maml_arch_base_path.exists():
                print(f"    Warning: New MAML base path {new_maml_arch_base_path} does not exist. Skipping new MAML loading for {arch_name}.")
            else:
                current_arch_maml_found = 0
                for seed_val in SEEDS: # SEEDS should be 0-9
                    expected_folder_pattern = f"exp_all_tasks_fomaml_{arch_name}_seed{seed_val}_*"
                    found_matching_folders = list(new_maml_arch_base_path.glob(expected_folder_pattern))
                    
                    if not found_matching_folders:
                        print(f"    No folder matching '{expected_folder_pattern}' found in {new_maml_arch_base_path} for seed {seed_val}.") # UNCOMMENTED
                        continue
                    if len(found_matching_folders) > 1:
                        print(f"    Warning: Multiple folders matching '{expected_folder_pattern}' found for seed {seed_val}. Using the first one: {found_matching_folders[0]}")
                    
                    maml_experiment_dir = found_matching_folders[0]
                    maml_model_path = maml_experiment_dir / 'best_model.pth' # Final weights

                    if maml_model_path.exists():
                        try:
                            state_dict_loaded = torch.load(maml_model_path, map_location=torch.device('cpu'), weights_only=True) # MODIFIED
                            state_dict_to_flatten = None
                            if isinstance(state_dict_loaded, dict) and 'model_state_dict' in state_dict_loaded:
                                state_dict_to_flatten = state_dict_loaded['model_state_dict']
                            elif isinstance(state_dict_loaded, dict) and 'state_dict' in state_dict_loaded: 
                                state_dict_to_flatten = state_dict_loaded['state_dict']
                            elif isinstance(state_dict_loaded, collections.OrderedDict) or isinstance(state_dict_loaded, dict): # Raw state_dict
                                state_dict_to_flatten = state_dict_loaded
                            else:
                                print(f"    Warning: Loaded MAML model for {arch_name}, seed {seed_val} from {maml_model_path} is not a state_dict or recognized checkpoint format. Type: {type(state_dict_loaded)}")

                            if state_dict_to_flatten:
                                if arch_name == 'conv4' and not printed_maml_summary_for_current_arch: # ADDED conditional print
                                    print_state_dict_summary(state_dict_to_flatten, f"NEW MAML {arch_name} (seed {seed_val})")
                                    printed_maml_summary_for_current_arch = True

                                flat_weights = flatten_weights(state_dict_to_flatten)
                                if flat_weights is not None:
                                    all_maml_weight_vectors.append(flat_weights)
                                    all_maml_point_labels.append(MAML_LABEL) 
                                    all_maml_seed_labels.append(seed_val)
                                    all_maml_point_categories.append('maml_all')
                                    maml_found_total_seeds += 1 
                                    current_arch_maml_found +=1
                        except Exception as e:
                            print(f"    Error loading NEW MAML (all tasks) weights for {arch_name}, seed {seed_val} from {maml_model_path}: {e}")
                print(f"    Found {current_arch_maml_found} NEW MAML (all tasks) models for {arch_name} from subdir '{NEW_MAML_RESULTS_SUBDIR}'.")
        else:
            print(f"  Skipping NEW MAML (all tasks) loading for {arch_name} as --maml_runs_base_dir ('{args.maml_runs_base_dir}') or NEW_MAML_RESULTS_SUBDIR ('{NEW_MAML_RESULTS_SUBDIR}') is not specified/valid.")

        # --- OLD MAML (trained on all tasks) weights loading (Commented Out) ---
        # \'\'\'
        # if arch_name in MAML_ALL_TASKS_PARENT_DIRS:
        #     print(f"  Loading MAML (all tasks) weights for {arch_name}...")
        #     maml_parent_dir_name = MAML_ALL_TASKS_PARENT_DIRS[arch_name]
        #     maml_arch_base_path = maml_runs_base_dir / maml_parent_dir_name if maml_runs_base_dir else Path(maml_parent_dir_name)
            
        #     filename_pattern = MAML_MODEL_FILENAME_PATTERNS[arch_name]
        #     # maml_found_total_seeds = 0 # Already initialized above

        #     for seed in SEEDS:
        #         model_file_relative_path = filename_pattern.format(seed=seed)
        #         maml_model_path = maml_arch_base_path / model_file_relative_path
        #         # print(f"    Attempting to find MAML model at: {maml_model_path}") 

        #         if maml_model_path.exists():
        #             try:
        #                 checkpoint = torch.load(maml_model_path, map_location=torch.device('cpu'))
        #                 if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        #                     state_dict = checkpoint['model_state_dict']
        #                 elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint: 
        #                     state_dict = checkpoint['state_dict']
        #                 else: 
        #                     state_dict = checkpoint
                        
        #                 flat_weights = flatten_weights(state_dict)
        #                 if flat_weights is not None:
        #                     all_maml_weight_vectors.append(flat_weights)
        #                     all_maml_point_labels.append(MAML_LABEL) 
        #                     all_maml_seed_labels.append(seed)
        #                     all_maml_point_categories.append('maml_all')
        #                     maml_found_total_seeds += 1
        #             except Exception as e:
        #                 print(f"    Error loading MAML (all tasks) weights for {arch_name}, seed {seed}: {e}")
        #     print(f"    Found {maml_found_total_seeds} MAML (all tasks) models for {arch_name}.")
        # else:
        #     print(f"  No MAML (all tasks) configuration defined for {arch_name}.")
        # \'\'\'
        # --- End of OLD MAML Loading ---

        # --- Placeholder for VANILLA (trained on all tasks) weights loading ---
        # ...

        # --- Combine all weights for PCA ---
        all_weights_for_pca = all_final_single_task_weight_vectors + all_initial_single_task_weight_vectors + all_maml_weight_vectors
        all_labels_for_pca = all_final_single_task_point_labels + all_initial_single_task_point_labels + all_maml_point_labels
        all_categories_for_pca = all_final_single_task_point_categories + all_initial_single_task_point_categories + all_maml_point_categories
        all_seeds_for_pca = all_final_single_task_seed_labels + all_initial_single_task_seed_labels + all_maml_seed_labels
        
        # print("\\n!!! --- NOTE: Performing PCA on INITIAL single-task weights ONLY for this run. --- !!!") # Removed this note
        
        total_vectors_for_pca = len(all_weights_for_pca)

        if total_vectors_for_pca < 2:
            print(f"  Skipping PCA for {arch_name}: only found {total_vectors_for_pca} total weight vector(s) across all categories. Need at least 2.")
            continue
        
        # Check if all weight vectors have the same shape
        first_vector_len = -1
        if all_weights_for_pca: # Check if list is not empty
            first_vector_len = len(all_weights_for_pca[0])
            all_same_len = all(len(vec) == first_vector_len for vec in all_weights_for_pca)
        else: # Should be caught by total_vectors_for_pca < 2, but for safety
            all_same_len = True 

        if not all_same_len:
            print(f"  Skipping PCA for {arch_name}: Not all collected weight vectors (initial, final, MAML) have the same length. First vector length: {first_vector_len}.")
            # Detailed debug print for mismatched lengths
            print(f"    Example lengths for {arch_name}:")
            counts = {'single_task_initial': 0, 'single_task_final': 0, 'maml_all': 0}
            max_prints_per_category = 2
            for i, vec in enumerate(all_weights_for_pca):
                category = all_categories_for_pca[i]
                if counts.get(category, 0) < max_prints_per_category:
                    label = all_labels_for_pca[i]
                    seed = all_seeds_for_pca[i]
                    print(f"      {category} ({label}, seed {seed}): {len(vec)}")
                    counts[category] = counts.get(category, 0) + 1
                if all(c >= max_prints_per_category for c in counts.values() if c is not None): # Check if all relevant categories hit max prints
                    if counts.get('maml_all', max_prints_per_category) >= (max_prints_per_category if all_maml_weight_vectors else 0) : # ensure maml prints if maml exists
                         break
            
            # Clear lists for the next architecture to prevent issues
            # Need to ensure all these combined lists are cleared or not used if PCA is skipped.
            # The 'continue' skips the rest of the loop for this architecture.
            # Let's ensure deletion of the combined lists as well.
            del all_final_single_task_weight_vectors, all_final_single_task_point_labels, all_final_single_task_seed_labels, all_final_single_task_point_categories
            del all_initial_single_task_weight_vectors, all_initial_single_task_point_labels, all_initial_single_task_seed_labels, all_initial_single_task_point_categories
            del all_maml_weight_vectors, all_maml_point_labels, all_maml_seed_labels, all_maml_point_categories
            del all_weights_for_pca, all_labels_for_pca, all_categories_for_pca, all_seeds_for_pca
            if 'X' in locals(): del X # X might not be defined if all_same_len is false early
            if 'principal_components' in locals(): del principal_components
            continue

        print(f"  Performing PCA on {total_vectors_for_pca} total weight vectors for {arch_name} (all same length: {first_vector_len})...")
        
        # --- AGGRESSIVE DEBUG: Print info about vectors going into PCA ---
        print(f"    DEBUG: About to stack vectors for PCA for {arch_name}. Samples:")
        max_debug_samples = 5 # Reduced max_debug_samples
        initial_printed = 0
        final_printed = 0
        maml_printed = 0 # Added for MAML
        for i in range(min(total_vectors_for_pca, max_debug_samples * 3)): # Check a decent number
            category = all_categories_for_pca[i]
            if category == 'single_task_initial' and initial_printed < max_debug_samples:
                print(f"      PCA Input (Initial) - Index {i}: Task '{all_labels_for_pca[i]}', Seed {all_seeds_for_pca[i]}, Norm: {np.linalg.norm(all_weights_for_pca[i])}")
                initial_printed += 1
            elif category == 'single_task_final' and final_printed < max_debug_samples:
                print(f"      PCA Input (Final)   - Index {i}: Task '{all_labels_for_pca[i]}', Seed {all_seeds_for_pca[i]}, Norm: {np.linalg.norm(all_weights_for_pca[i])}")
                final_printed += 1
            elif category == 'maml_all' and maml_printed < max_debug_samples: # Added MAML debug
                print(f"      PCA Input (MAML)    - Index {i}: Label '{all_labels_for_pca[i]}', Seed {all_seeds_for_pca[i]}, Norm: {np.linalg.norm(all_weights_for_pca[i])}")
                maml_printed +=1
            if initial_printed >= max_debug_samples and final_printed >=max_debug_samples and maml_printed >= max_debug_samples: 
                 break
        if total_vectors_for_pca > (max_debug_samples * 3) and not (initial_printed >= max_debug_samples and final_printed >=max_debug_samples and maml_printed >=max_debug_samples) :
             print("      ... (more samples not printed)")
        print("    ----------------------------------------------------------")
        # --- END AGGRESSIVE DEBUG ---\n
        try:
            X = np.stack(all_weights_for_pca)
            
            # Apply PCA
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(X)
            
            plt.figure(figsize=(14, 12))
            
            # --- Plot INITIAL and FINAL single-task model points & lines connecting them ---
            for task_name in PB_TASKS: # Iterate through tasks first for cleaner legend grouping potentially
                # Collect all points for this task to plot together (initial, final)
                # This helps if we want to ensure lines are drawn correctly and points are from same seed.
                
                # Initial points for this task
                task_initial_indices = [i for i, (label, category) in enumerate(zip(all_labels_for_pca, all_categories_for_pca))
                                        if category == 'single_task_initial' and label == task_name]
                if task_initial_indices:
                    plt.scatter(principal_components[task_initial_indices, 0],
                                principal_components[task_initial_indices, 1],
                                color=task_colors[task_name],
                                marker='x', s=50, alpha=0.7) # No individual label here, use collective legend

                # Final points for this task
                task_final_indices = [i for i, (label, category) in enumerate(zip(all_labels_for_pca, all_categories_for_pca))
                                      if category == 'single_task_final' and label == task_name]
                if task_final_indices:
                    plt.scatter(principal_components[task_final_indices, 0],
                                principal_components[task_final_indices, 1],
                                color=task_colors[task_name],
                                marker='o', s=60, alpha=0.8) # No individual label here

                # Lines connecting initial to final for this task (seed by seed)
                for original_seed_val in SEEDS:
                    initial_idx_list = [i for i, (label, cat, seed_lbl) in 
                                        enumerate(zip(all_labels_for_pca, all_categories_for_pca, all_seeds_for_pca))
                                        if cat == 'single_task_initial' and label == task_name and seed_lbl == original_seed_val]
                    final_idx_list = [i for i, (label, cat, seed_lbl) in 
                                      enumerate(zip(all_labels_for_pca, all_categories_for_pca, all_seeds_for_pca))
                                      if cat == 'single_task_final' and label == task_name and seed_lbl == original_seed_val]

                    if initial_idx_list and final_idx_list:
                        initial_idx = initial_idx_list[0]
                        final_idx = final_idx_list[0]
                        plt.plot([principal_components[initial_idx, 0], principal_components[final_idx, 0]],
                                 [principal_components[initial_idx, 1], principal_components[final_idx, 1]],
                                 color=task_colors[task_name], linestyle='-', linewidth=0.5, alpha=0.5)

            # --- Plot MAML (all tasks) model points ---
            maml_indices = [i for i, category in enumerate(all_categories_for_pca) if category == 'maml_all']
            if maml_indices:
                plt.scatter(principal_components[maml_indices, 0],
                            principal_components[maml_indices, 1],
                            color=maml_plot_style['color'],
                            marker=maml_plot_style['marker'],
                            s=maml_plot_style['s'],
                            # label=maml_plot_style['label'] + f" (N={len(maml_indices)})", # Label handled by legend_handles
                            alpha=maml_plot_style['alpha'],
                            zorder=maml_plot_style['zorder'])
            
            plt.title(f'PCA of {arch_name} Weights (Single-Task & Meta-Trained)')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.grid(True)
            
            # --- Create Legend ---
            legend_handles = []
            unique_tasks_in_plot = sorted(list(set(all_labels_for_pca) - {MAML_LABEL}))

            for task_name_legend in unique_tasks_in_plot:
                if task_name_legend not in PB_TASKS: continue # Ensure it's a known task for color mapping

                # Final single-task
                if any(label == task_name_legend and cat == 'single_task_final' for label, cat in zip(all_labels_for_pca, all_categories_for_pca)):
                    legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label=f"{task_name_legend} (Final)",
                                              markerfacecolor=task_colors[task_name_legend], markersize=8))
                # Initial single-task
                if any(label == task_name_legend and cat == 'single_task_initial' for label, cat in zip(all_labels_for_pca, all_categories_for_pca)):
                    legend_handles.append(plt.Line2D([0], [0], marker='x', color=task_colors[task_name_legend], label=f"{task_name_legend} (Initial)",
                                              linestyle='None', markersize=8)) # markerfacecolor not great for 'x', use color

            if maml_indices: # Check if maml_indices has been defined from plotting
                legend_handles.append(plt.Line2D([0], [0], marker=maml_plot_style['marker'], color='w',
                                          label=maml_plot_style['label'] + f" (N={len(maml_indices)})",
                                          markerfacecolor=maml_plot_style['color'], markersize=10))
            
            plt.legend(handles=legend_handles, title="Model Weights", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

            # Save plot
            plot_filename = output_plot_dir / f'pca_weights_combined_{arch_name}.png'
            plt.savefig(plot_filename)
            plt.close() 
            print(f"  PCA plot saved to {plot_filename}")

        except Exception as e:
            print(f"  Error during PCA or plotting for {arch_name}: {e}")
        
        # Clear lists for the next architecture
        del all_final_single_task_weight_vectors, all_final_single_task_point_labels, all_final_single_task_seed_labels, all_final_single_task_point_categories
        del all_initial_single_task_weight_vectors, all_initial_single_task_point_labels, all_initial_single_task_seed_labels, all_initial_single_task_point_categories
        del all_maml_weight_vectors, all_maml_point_labels, all_maml_seed_labels, all_maml_point_categories
        del all_weights_for_pca, all_labels_for_pca, all_categories_for_pca, all_seeds_for_pca
        if 'X' in locals(): del X
        if 'principal_components' in locals(): del principal_components

    print("\nAnalysis finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform PCA on saved model weights across all PB tasks, MAML (all tasks), and Vanilla (all tasks).')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Base directory containing the SINGLE-TASK results (e.g., results/pb_single_task)')
    parser.add_argument('--maml_runs_base_dir', type=str, default=None, # New argument
                        help='Base directory where MAML multi-task model parent folders (like exp1_...conv6lr...) are located. '
                             'If MAML_ALL_TASKS_PARENT_DIRS contains absolute paths, this can be omitted or ignored. '
                             'The script will look for a subdirectory defined by NEW_MAML_RESULTS_SUBDIR within this base directory.')
    # We will add --vanilla_runs_base_dir later
    parser.add_argument('--output_plot_dir', type=str, default='results/pca_plots_combined',
                        help='Directory to save the generated PCA plots.')
    
    args = parser.parse_args()
    main(args) 