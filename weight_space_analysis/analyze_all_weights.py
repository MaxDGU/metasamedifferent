import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
import sys
import collections # For defaultdict

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import model definitions
from baselines.models.conv2 import SameDifferentCNN as Conv2CNN
from baselines.models.conv4 import SameDifferentCNN as Conv4CNN
from baselines.models.conv6 import SameDifferentCNN as Conv6CNN

# --- Constants (should match the run script) ---
ARCHITECTURES = {
    'conv2': Conv2CNN,
    'conv4': Conv4CNN,
    'conv6': Conv6CNN
}
SEEDS = list(range(10)) # Seeds 0 through 9 (original general list)
PB_TASKS = ['regular', 'lines', 'open', 'wider_line', 'scrambled',
            'random_color', 'arrows', 'irregular', 'filled', 'original']

# --- Paths for NEW MAML models (trained on all tasks, from newer experiments) ---
NEW_MAML_RESULTS_SUBDIR = "maml_results_new_2" 

MAML_LABEL = "MAML (all tasks)"
INITIAL_LABEL = "Initial Weights"

def flatten_weights(state_dict):
    all_weights = []
    if state_dict is None: 
        print("    DEBUG: flatten_weights received a None state_dict.")
        return None
    for key in sorted(state_dict.keys()):
        param = state_dict[key]
        if isinstance(param, torch.Tensor):
            all_weights.append(param.detach().cpu().numpy().flatten())
    if not all_weights:
        print("    DEBUG: flatten_weights: no weights found in state_dict to concatenate.")
        return None
    concatenated = np.concatenate(all_weights)
    return concatenated

def print_state_dict_summary(state_dict, model_type_label):
    print(f"    DEBUG: State dict summary for {model_type_label}:")
    total_params = 0
    keys_found = 0
    if state_dict is None:
        print(f"    DEBUG: State dict for {model_type_label} is None.")
        return
    for k, v in sorted(state_dict.items()):
        keys_found +=1
        if isinstance(v, torch.Tensor):
            # print(f"      Key: {k}, Shape: {v.shape}, Numel: {v.numel()}") # Verbose
            total_params += v.numel()
        # else:
            # print(f"      Key: {k}, Type: {type(v)}") # Verbose
    if keys_found == 0:
        print(f"    DEBUG: State dict for {model_type_label} is empty.")
    print(f"    DEBUG: Total parameters in {model_type_label} state_dict: {total_params}")

def main(args):
    base_results_dir = Path(args.results_dir)
    maml_runs_base_dir = Path(args.maml_runs_base_dir) if args.maml_runs_base_dir else None
    output_plot_dir = Path(args.output_plot_dir)
    output_plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing weights.")
    print(f"Single-Task Results Dir: {base_results_dir}")
    if args.specific_maml_experiment_path:
        print(f"Specific MAML Experiment Path: {args.specific_maml_experiment_path}")
    elif maml_runs_base_dir and NEW_MAML_RESULTS_SUBDIR:
        print(f"General MAML models from subdir '{NEW_MAML_RESULTS_SUBDIR}' under: {maml_runs_base_dir}")
    print(f"Saving plots to: {output_plot_dir}")

    cmap = plt.get_cmap('tab10') 
    task_colors = {task: cmap(i) for i, task in enumerate(PB_TASKS)}
    maml_plot_style = {'color': 'black', 'marker': '*', 's': 200, 'label': MAML_LABEL, 'alpha': 0.9, 'zorder': 10}
    initial_plot_style = {'color': 'grey', 'marker': 'D', 's': 100, 'label': INITIAL_LABEL, 'alpha': 0.8, 'zorder': 11}

    # Determine architectures and seeds to process
    architectures_to_process = {args.specific_arch_to_analyze: ARCHITECTURES[args.specific_arch_to_analyze]} if args.specific_arch_to_analyze else ARCHITECTURES
    seeds_to_process = [args.specific_seed_to_analyze] if args.specific_seed_to_analyze is not None else SEEDS
    num_original_seeds_for_calc = len(SEEDS) # Used for globally_unique_seed calculation in general mode

    if args.specific_arch_to_analyze or args.specific_seed_to_analyze:
        print(f"Running in SPECIFIC mode for Arch: {args.specific_arch_to_analyze or 'All'}, Seed: {args.specific_seed_to_analyze if args.specific_seed_to_analyze is not None else 'All'}")

    for arch_name, model_class in architectures_to_process.items():
        print(f"\n===== Processing Architecture: {arch_name} ====")
        printed_final_single_task_summary_for_current_arch = False
        printed_maml_summary_for_current_arch = False

        all_final_single_task_weight_vectors = []
        all_final_single_task_point_labels = [] 
        all_final_single_task_seed_labels = []  
        all_final_single_task_point_categories = [] 

        # For the ideal experiment, there's only ONE initial weight vector for the specific seed
        initial_weight_vector_for_specific_run = None 
        initial_weight_label_for_specific_run = None
        initial_weight_seed_for_specific_run = None
        initial_weight_category_for_specific_run = None

        all_maml_weight_vectors = []
        all_maml_point_labels = []
        all_maml_seed_labels = []
        all_maml_point_categories = []
        
        found_final_weights = 0
        maml_found_total_seeds = 0

        print(f"  Loading SINGLE-TASK (final) weights for {arch_name}...")
        single_task_seed_to_use = args.specific_seed_to_analyze if args.specific_seed_to_analyze is not None else None

        for task_idx, task_name in enumerate(PB_TASKS):
            task_found_final_seeds_this_task = 0 # Renamed for clarity
            for seed_val_to_load in seeds_to_process: # This will be just one seed if specific_seed is given
                
                globally_unique_seed_for_folder = seed_val_to_load # For specific seed run, folder is just seed_X
                if not args.specific_seed_to_analyze: # General case, calculate from original SEEDS list
                    # This part is for the original script logic when not in specific mode
                    original_seed_run_idx = SEEDS.index(seed_val_to_load) if seed_val_to_load in SEEDS else -1
                    if original_seed_run_idx == -1: continue # Should not happen if seeds_to_process is from SEEDS
                    globally_unique_seed_for_folder = (task_idx * num_original_seeds_for_calc) + seed_val_to_load
                
                model_folder_path = base_results_dir / task_name / arch_name / f'seed_{globally_unique_seed_for_folder}'
                final_model_path = model_folder_path / 'best_model.pth'
                initial_model_path_candidate = model_folder_path / 'initial_model.pth' # Candidate initial model

                # Load INITIAL weights (only ONCE for specific seed run)
                if args.specific_seed_to_analyze is not None and initial_weight_vector_for_specific_run is None:
                    if initial_model_path_candidate.exists():
                        try:
                            print(f"    Attempting to load THE initial model from: {initial_model_path_candidate}")
                            state_dict_initial = torch.load(initial_model_path_candidate, map_location=torch.device('cpu'), weights_only=True)
                            flat_weights_initial = flatten_weights(state_dict_initial)
                            if flat_weights_initial is not None:
                                initial_weight_vector_for_specific_run = flat_weights_initial
                                initial_weight_label_for_specific_run = INITIAL_LABEL 
                                initial_weight_seed_for_specific_run = seed_val_to_load 
                                initial_weight_category_for_specific_run = 'initial_common'
                                print(f"    Successfully loaded and flattened THE initial model (Seed {seed_val_to_load}). Norm: {np.linalg.norm(flat_weights_initial):.4e}")
                                if arch_name == args.specific_arch_to_analyze or (not args.specific_arch_to_analyze and arch_name == 'conv4'):
                                     print_state_dict_summary(state_dict_initial, f"INITIAL common ({arch_name}, seed {seed_val_to_load})")
                            else:
                                print(f"    Warning: Flatten_weights returned None for THE initial model at {initial_model_path_candidate}")
                        except Exception as e:
                            print(f"    Error loading THE initial model from {initial_model_path_candidate}: {e}")
                    else:
                        print(f"    Warning: THE initial model path NOT FOUND: {initial_model_path_candidate} (will not plot initial point if not found from any task dir for this seed)")
                
                # Load FINAL weights (best_model.pth)
                if final_model_path.exists():
                    try:
                        state_dict = torch.load(final_model_path, map_location=torch.device('cpu'), weights_only=True)
                        if arch_name == 'conv4' and not printed_final_single_task_summary_for_current_arch and not args.specific_seed_to_analyze : # Debug for general case
                            print_state_dict_summary(state_dict, f"FINAL single-task {arch_name} ({task_name}, seed {seed_val_to_load})")
                            printed_final_single_task_summary_for_current_arch = True

                        flat_weights = flatten_weights(state_dict)
                        if flat_weights is not None:
                            all_final_single_task_weight_vectors.append(flat_weights)
                            all_final_single_task_point_labels.append(task_name)
                            all_final_single_task_seed_labels.append(seed_val_to_load)
                            all_final_single_task_point_categories.append('single_task_final')
                            print(f"      Loaded FINAL single-task: {task_name}, seed {seed_val_to_load}, norm: {np.linalg.norm(flat_weights):.4e}")
                            found_final_weights += 1
                            task_found_final_seeds_this_task += 1
                    except Exception as e:
                        print(f"    Error loading FINAL single-task weights for {arch_name}, task {task_name}, seed {seed_val_to_load} (folder seed_{globally_unique_seed_for_folder}) from {final_model_path}: {e}")
            # End seed loop
            if task_found_final_seeds_this_task > 0:
                 print(f"    Found {task_found_final_seeds_this_task} final models for task '{task_name}' (Seed(s): {seeds_to_process})")
        # End task loop
        print(f"  Finished loading single-task final weights for {arch_name}. Found {found_final_weights} total FINAL vectors.")
        if args.specific_seed_to_analyze is not None and initial_weight_vector_for_specific_run is None:
            print(f"  CRITICAL WARNING: For specific seed run (seed {args.specific_seed_to_analyze}), THE initial model was NOT loaded. PCA will be missing the initial point.")

        # --- Load MAML (all tasks) weights for the current architecture ---
        if args.specific_maml_experiment_path:
            print(f"  Loading SPECIFIC MAML (all tasks) model for {arch_name} from path: {args.specific_maml_experiment_path}")
            maml_model_path = Path(args.specific_maml_experiment_path) / 'best_model.pth'
            maml_initial_path_candidate = Path(args.specific_maml_experiment_path) / 'initial_model.pth'

            # Try to load initial from MAML dir if not already loaded (for specific seed run consistency)
            if args.specific_seed_to_analyze is not None and initial_weight_vector_for_specific_run is None and maml_initial_path_candidate.exists():
                try:
                    print(f"    Attempting to load THE initial model from MAML dir: {maml_initial_path_candidate}")
                    state_dict_initial = torch.load(maml_initial_path_candidate, map_location=torch.device('cpu'), weights_only=True)
                    flat_weights_initial = flatten_weights(state_dict_initial)
                    if flat_weights_initial is not None:
                        initial_weight_vector_for_specific_run = flat_weights_initial
                        initial_weight_label_for_specific_run = INITIAL_LABEL 
                        initial_weight_seed_for_specific_run = args.specific_seed_to_analyze 
                        initial_weight_category_for_specific_run = 'initial_common'
                        print(f"    Successfully loaded and flattened THE initial model (Seed {args.specific_seed_to_analyze}) from MAML dir. Norm: {np.linalg.norm(flat_weights_initial):.4e}")
                        if arch_name == args.specific_arch_to_analyze or (not args.specific_arch_to_analyze and arch_name == 'conv4'):
                             print_state_dict_summary(state_dict_initial, f"INITIAL common from MAML dir ({arch_name}, seed {args.specific_seed_to_analyze})")
                    else: print(f"    Warning: Flatten_weights returned None for THE initial model from MAML dir {maml_initial_path_candidate}")
                except Exception as e: print(f"    Error loading THE initial model from MAML dir {maml_initial_path_candidate}: {e}")

            if maml_model_path.exists():
                try:
                    state_dict_loaded = torch.load(maml_model_path, map_location=torch.device('cpu'), weights_only=True)
                    state_dict_to_flatten = None
                    if isinstance(state_dict_loaded, dict) and 'model_state_dict' in state_dict_loaded:
                        state_dict_to_flatten = state_dict_loaded['model_state_dict']
                    elif isinstance(state_dict_loaded, dict) and 'state_dict' in state_dict_loaded: 
                        state_dict_to_flatten = state_dict_loaded['state_dict']
                    elif isinstance(state_dict_loaded, collections.OrderedDict) or isinstance(state_dict_loaded, dict):
                        state_dict_to_flatten = state_dict_loaded
                    else:
                        print(f"    Warning: Loaded MAML model from {maml_model_path} is not a state_dict or recognized checkpoint format. Type: {type(state_dict_loaded)}")

                    if state_dict_to_flatten:
                        if arch_name == args.specific_arch_to_analyze or (not args.specific_arch_to_analyze and arch_name == 'conv4'):
                             print_state_dict_summary(state_dict_to_flatten, f"SPECIFIC MAML {arch_name} (seed {args.specific_seed_to_analyze if args.specific_seed_to_analyze is not None else 'N/A'})")
                             printed_maml_summary_for_current_arch = True
                        flat_weights = flatten_weights(state_dict_to_flatten)
                        if flat_weights is not None:
                            all_maml_weight_vectors.append(flat_weights)
                            all_maml_point_labels.append(MAML_LABEL) 
                            all_maml_seed_labels.append(args.specific_seed_to_analyze if args.specific_seed_to_analyze is not None else -1) # Use specific seed or -1
                            all_maml_point_categories.append('maml_all')
                            maml_found_total_seeds += 1 # Counts as one MAML model for this specific run
                            print(f"    Successfully loaded and flattened SPECIFIC MAML model. Norm: {np.linalg.norm(flat_weights):.4e}")
                        else:
                            print(f"    Warning: Flatten_weights returned None for SPECIFIC MAML model from {maml_model_path}")
                    else:
                        print(f"    Warning: state_dict_to_flatten is None for MAML model at {maml_model_path}. Cannot process.")
                except Exception as e:
                    print(f"    Error loading SPECIFIC MAML (all tasks) weights for {arch_name} from {maml_model_path}: {e}")
            print(f"    Found {maml_found_total_seeds} SPECIFIC MAML (all tasks) model for {arch_name}.")
        
        elif maml_runs_base_dir and NEW_MAML_RESULTS_SUBDIR: # Original general MAML loading logic
            print(f"  Loading NEW MAML (all tasks) weights for {arch_name} from subdir '{NEW_MAML_RESULTS_SUBDIR}'...")
            new_maml_arch_base_path = maml_runs_base_dir / NEW_MAML_RESULTS_SUBDIR
            if not new_maml_arch_base_path.exists():
                print(f"    Warning: New MAML base path {new_maml_arch_base_path} does not exist. Skipping.")
            else:
                current_arch_maml_found = 0
                for seed_val in seeds_to_process: # General case: iterate through seeds_to_process
                    expected_folder_pattern = f"exp_all_tasks_fomaml_{arch_name}_seed{seed_val}_*"
                    found_matching_folders = list(new_maml_arch_base_path.glob(expected_folder_pattern))
                    if not found_matching_folders:
                        print(f"    No folder matching '{expected_folder_pattern}' found for seed {seed_val}.")
                        continue
                    if len(found_matching_folders) > 1:
                        print(f"    Warning: Multiple folders matching '{expected_folder_pattern}' for seed {seed_val}. Using first: {found_matching_folders[0]}")
                    maml_experiment_dir = found_matching_folders[0]
                    maml_model_path = maml_experiment_dir / 'best_model.pth'
                    if maml_model_path.exists():
                        try:
                            state_dict_loaded = torch.load(maml_model_path, map_location=torch.device('cpu'), weights_only=True)
                            state_dict_to_flatten = None # Reset for each model
                            if isinstance(state_dict_loaded, dict) and 'model_state_dict' in state_dict_loaded:
                                state_dict_to_flatten = state_dict_loaded['model_state_dict']
                            elif isinstance(state_dict_loaded, dict) and 'state_dict' in state_dict_loaded: 
                                state_dict_to_flatten = state_dict_loaded['state_dict']
                            elif isinstance(state_dict_loaded, collections.OrderedDict) or isinstance(state_dict_loaded, dict):
                                state_dict_to_flatten = state_dict_loaded
                            else: print(f"    Warning: Loaded MAML model for {arch_name}, seed {seed_val} from {maml_model_path} is not a recognized checkpoint. Type: {type(state_dict_loaded)}")
                            
                            if state_dict_to_flatten:
                                if arch_name == 'conv4' and not printed_maml_summary_for_current_arch: # Debug print
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
                        except Exception as e: print(f"    Error loading NEW MAML (all tasks) weights for {arch_name}, seed {seed_val} from {maml_model_path}: {e}")
                print(f"    Found {current_arch_maml_found} NEW MAML (all tasks) models for {arch_name} from subdir '{NEW_MAML_RESULTS_SUBDIR}'.")
        else:
            if not args.specific_maml_experiment_path: # Only print skip if not in specific MAML mode
                 print(f"  Skipping MAML (all tasks) loading for {arch_name} as relevant args not specified.")

        # --- Combine all weights for PCA --- 
        all_weights_for_pca = []
        all_labels_for_pca = []
        all_categories_for_pca = []
        all_seeds_for_pca = []

        # Add the single initial model if loaded for specific run
        if initial_weight_vector_for_specific_run is not None:
            all_weights_for_pca.append(initial_weight_vector_for_specific_run)
            all_labels_for_pca.append(initial_weight_label_for_specific_run)
            all_categories_for_pca.append(initial_weight_category_for_specific_run)
            all_seeds_for_pca.append(initial_weight_seed_for_specific_run)
            print(f"  Added THE initial model (Seed {initial_weight_seed_for_specific_run}) to PCA list.")
        
        all_weights_for_pca.extend(all_final_single_task_weight_vectors)
        all_labels_for_pca.extend(all_final_single_task_point_labels)
        all_categories_for_pca.extend(all_final_single_task_point_categories)
        all_seeds_for_pca.extend(all_final_single_task_seed_labels)
        
        all_weights_for_pca.extend(all_maml_weight_vectors)
        all_labels_for_pca.extend(all_maml_point_labels)
        all_categories_for_pca.extend(all_maml_point_categories)
        all_seeds_for_pca.extend(all_maml_seed_labels)
                
        total_vectors_for_pca = len(all_weights_for_pca)

        if total_vectors_for_pca < 2:
            print(f"  Skipping PCA for {arch_name}: only found {total_vectors_for_pca} total weight vector(s). Need at least 2.")
            continue
        
        first_vector_len = -1
        if all_weights_for_pca:
            first_vector_len = len(all_weights_for_pca[0])
            all_same_len = all(len(vec) == first_vector_len for vec in all_weights_for_pca)
        else: all_same_len = True 

        if not all_same_len:
            print(f"  Skipping PCA for {arch_name}: Not all collected weight vectors have the same length. First vector length: {first_vector_len}.")
            counts = {}
            max_prints_per_category = 3
            for i, vec in enumerate(all_weights_for_pca):
                category = all_categories_for_pca[i]
                label = all_labels_for_pca[i]
                seed = all_seeds_for_pca[i]
                current_cat_count = counts.get(category, 0)
                if current_cat_count < max_prints_per_category:
                    print(f"      {category} ({label}, seed {seed}): len={len(vec)}")
                    counts[category] = current_cat_count + 1
            # Clear lists for the next architecture to prevent issues - already done by re-init at loop start
            continue

        print(f"  Performing PCA on {total_vectors_for_pca} total weight vectors for {arch_name} (all same length: {first_vector_len})...")
        # --- AGGRESSIVE DEBUG (removed for brevity, was here) ---
        try:
            X = np.stack(all_weights_for_pca)
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(X)
            plt.figure(figsize=(14, 12))
            
            # --- Plot points based on category --- 
            plotted_initial_label = False
            plotted_maml_label = False

            # Create a dictionary to hold handles for task final points for legend
            task_final_legend_handles = {}

            for i in range(total_vectors_for_pca):
                pc1, pc2 = principal_components[i, 0], principal_components[i, 1]
                category = all_categories_for_pca[i]
                label = all_labels_for_pca[i]
                seed = all_seeds_for_pca[i]
                
                if category == 'initial_common':
                    plt.scatter(pc1, pc2, 
                                color=initial_plot_style['color'], 
                                marker=initial_plot_style['marker'], 
                                s=initial_plot_style['s'], 
                                # label=initial_plot_style['label'] if not plotted_initial_label else "", # Handled by collective legend
                                alpha=initial_plot_style['alpha'], 
                                zorder=initial_plot_style['zorder'])
                    plotted_initial_label = True
                elif category == 'maml_all':
                    plt.scatter(pc1, pc2, 
                                color=maml_plot_style['color'], 
                                marker=maml_plot_style['marker'], 
                                s=maml_plot_style['s'], 
                                # label=maml_plot_style['label'] if not plotted_maml_label else "", # Handled by collective legend
                                alpha=maml_plot_style['alpha'], 
                                zorder=maml_plot_style['zorder'])
                    plotted_maml_label = True
                elif category == 'single_task_final':
                    task_color = task_colors.get(label, 'grey') # Default to grey if task not in map
                    # Create handle for legend only once per task
                    if label not in task_final_legend_handles:
                        task_final_legend_handles[label] = plt.Line2D([0], [0], marker='o', color='w', 
                                                                     label=f"{label} (Final)", markerfacecolor=task_color, markersize=8)
                    plt.scatter(pc1, pc2, color=task_color, marker='o', s=60, alpha=0.8)
            
            # --- Lines connecting initial to final (only for specific seed run) ---
            if args.specific_seed_to_analyze is not None and initial_weight_vector_for_specific_run is not None:
                initial_idx_in_pca = -1
                try:
                    initial_idx_in_pca = all_categories_for_pca.index('initial_common')
                except ValueError:
                    pass # initial_common not found, should not happen if added
                
                if initial_idx_in_pca != -1:
                    initial_pc1 = principal_components[initial_idx_in_pca, 0]
                    initial_pc2 = principal_components[initial_idx_in_pca, 1]
                    for i in range(total_vectors_for_pca):
                        if all_categories_for_pca[i] == 'single_task_final' and all_seeds_for_pca[i] == args.specific_seed_to_analyze:
                            final_pc1 = principal_components[i, 0]
                            final_pc2 = principal_components[i, 1]
                            task_label_for_line = all_labels_for_pca[i]
                            line_color = task_colors.get(task_label_for_line, 'grey')
                            plt.plot([initial_pc1, final_pc1], [initial_pc2, final_pc2],
                                     color=line_color, linestyle='-', linewidth=0.7, alpha=0.6)

            title_suffix = f"(Seed {args.specific_seed_to_analyze})" if args.specific_seed_to_analyze is not None else "(All Seeds)"
            plt.title(f'PCA of {arch_name} Weights {title_suffix}')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.grid(True)
            
            legend_handles = []
            if initial_weight_vector_for_specific_run is not None or any(cat == 'initial_common' for cat in all_categories_for_pca):
                 legend_handles.append(plt.Line2D([0], [0], marker=initial_plot_style['marker'], color='w', 
                                           label=initial_plot_style['label'], 
                                           markerfacecolor=initial_plot_style['color'], markersize=10))
            
            # Add sorted task final handles
            for task_name_legend in sorted(task_final_legend_handles.keys()):
                legend_handles.append(task_final_legend_handles[task_name_legend])

            if any(cat == 'maml_all' for cat in all_categories_for_pca):
                num_maml_points = sum(1 for cat in all_categories_for_pca if cat == 'maml_all')
                legend_handles.append(plt.Line2D([0], [0], marker=maml_plot_style['marker'], color='w',
                                          label=maml_plot_style['label'] + f" (N={num_maml_points})",
                                          markerfacecolor=maml_plot_style['color'], markersize=10))
            
            plt.legend(handles=legend_handles, title="Model Weights", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout(rect=[0, 0, 0.85, 1])

            plot_filename_suffix = f"_seed{args.specific_seed_to_analyze}" if args.specific_seed_to_analyze is not None else "_all_seeds"
            plot_filename = output_plot_dir / f'pca_weights_combined_{arch_name}{plot_filename_suffix}.png'
            plt.savefig(plot_filename)
            plt.close() 
            print(f"  PCA plot saved to {plot_filename}")

        except Exception as e:
            print(f"  Error during PCA or plotting for {arch_name}: {e}")
            import traceback
            traceback.print_exc() # ADDED for more detail
        
        # Clear lists for the next architecture (already re-initialized at start of loop)

    print("\nAnalysis finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform PCA on model weights.')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Base directory for SINGLE-TASK results (e.g., results/pb_single_task or results/ideal_exp/single_task_runs)')
    parser.add_argument('--maml_runs_base_dir', type=str, default=None, 
                        help='Base directory for general MAML experiment folders (e.g., results/pb_maml_Svar_Q3_runs). Used if not in specific MAML mode.')
    parser.add_argument('--output_plot_dir', type=str, default='results/pca_plots_combined',
                        help='Directory to save the generated PCA plots.')
    
    # New arguments for specific (ideal) experiment run
    parser.add_argument('--specific_arch_to_analyze', type=str, default=None, choices=list(ARCHITECTURES.keys()),
                        help='If specified, only analyze this architecture.')
    parser.add_argument('--specific_seed_to_analyze', type=int, default=None,
                        help='If specified, only analyze for this specific seed (e.g., for ideal experiment runs).')
    parser.add_argument('--specific_maml_experiment_path', type=str, default=None,
                        help='Full path to a specific MAML experiment directory (e.g., for ideal experiment run). Overrides general MAML search.')

    args = parser.parse_args()
    main(args) 