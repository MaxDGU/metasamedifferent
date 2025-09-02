# naturalistic/meta_train/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import random
import sys # Added for path adjustment
import gc # Added for memory management
import learn2learn as l2l # Added for MAML

# Add project root to sys.path to allow for 'from baselines.models...' imports
# Assumes the script is in <project_root>/naturalistic/
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# --- Model Imports ---
# Script and models are assumed to be in the same directory or on PYTHONPATH
# No complex sys.path manipulation needed if running from the directory containing these files.
# script_dir = Path(__file__).parent.resolve()
# project_root = script_dir.parent.parent.parent # This was based on a different assumed structure
# sys.path.append(str(project_root))
# print(f"Added project root to sys.path: {project_root}")

try:
    # Import specific model classes directly
    from meta_baseline.models.conv2lr import SameDifferentCNN as Conv2CNN
    from meta_baseline.models.conv4lr import SameDifferentCNN as Conv4CNN
    from meta_baseline.models.conv6lr import SameDifferentCNN as Conv6CNN
    print("Successfully imported Conv{2,4,6}lrCNN models from baselines.models")

    # # Attempt to import utils_meta - might fail, handle later
    # try:
    #     from meta_baseline.utils_meta import train_epoch, validate 
    #     print("Successfully imported train_epoch/validate from utils_meta")
    #     UTILS_META_AVAILABLE = True
    # except ImportError:
    #     print("Could not import from utils_meta. Will implement MAML steps manually.")
    #     UTILS_META_AVAILABLE = False
    UTILS_META_AVAILABLE = False # Assume not available for now

except ImportError as e:
    print(f"Error importing models (direct import): {e}")
    print("Please ensure conv2.py, conv4.py, conv6.py are in the same directory as this script or accessible via PYTHONPATH.")
    exit(1)

# Define ARCHITECTURES dictionary locally
ARCHITECTURES = {
    'conv2lr': Conv2CNN,
    'conv4lr': Conv4CNN,
    'conv6lr': Conv6CNN
}

# --- Accuracy Function (from reference script) ---
def accuracy(predictions, targets):
    """Binary accuracy: assumes predictions are logits [N, 2], targets are [N]."""
    # Get the index of the max logit (class 1 if predictions[:, 1] > predictions[:, 0])
    predicted_labels = torch.argmax(predictions, dim=1)
    correct = (predicted_labels == targets).float()
    return correct.mean()

# --- Data Loading (MetaNaturalisticDataset class remains the same) ---
class MetaNaturalisticDataset(Dataset):
    """
    Dataset for loading meta-learning episodes from the naturalistic HDF5 files.
    Reads episodes stored in individual groups (e.g., 'episode_000000').
    Images will be resized to 128x128 and Imagenet normalized.
    """
    def __init__(self, h5_path, transform=None):
        self.h5_path = Path(h5_path)
        self.user_transform = transform # Store user-provided transform

        # Default transforms including resizing and Imagenet normalization
        from torchvision import transforms as T 
        self.processing_transform = T.Compose([
            # Input to this transform will be a single HWC uint8 numpy array image
            T.ToPILImage(), 
            T.Resize((128, 128)),
            T.ToTensor(), # Converts PIL (0-255) to FloatTensor (0-1) and C,H,W
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        self._file = None 
        self.episode_keys = []

        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        try:
            self._file = h5py.File(self.h5_path, 'r')
            self.episode_keys = sorted([k for k in self._file.keys() if k.startswith('episode_')])
            if not self.episode_keys:
                raise ValueError(f"No episode groups found in {self.h5_path}")
            print(f"Loaded {len(self.episode_keys)} episodes from {self.h5_path.name}")
        except Exception as e:
            print(f"Error opening or reading keys from {self.h5_path}: {e}")
            if self._file:
                self._file.close()
            raise

    def __len__(self):
        return len(self.episode_keys)

    def __getitem__(self, idx):
        if not hasattr(self, '_file') or not self._file:
             self._file = h5py.File(self.h5_path, 'r')

        episode_key = self.episode_keys[idx]
        try:
            ep_group = self._file[episode_key]
            support_images = ep_group['support_images'][()] # [S, H, W, C] uint8
            support_labels = ep_group['support_labels'][()] # [S] int32
            query_images = ep_group['query_images'][()]     # [Q, H, W, C] uint8
            query_labels = ep_group['query_labels'][()]     # [Q] int32
        except KeyError as e:
            print(f"Error accessing data for key {episode_key} in {self.h5_path}: {e}")
            # Return dummy data or raise error, depending on desired handling
            # For now, re-raise to indicate a problem
            raise
        except Exception as e:
            print(f"Unexpected error reading episode {episode_key}: {e}")
            raise

        # --- Apply Transforms ---
        # support_images is (S, H, W, C) uint8 numpy array
        # query_images is (Q, H, W, C) uint8 numpy array

        transformed_support_images = []
        for i in range(support_images.shape[0]):
            # Pass each image (H, W, C) uint8 numpy array through the processing transform pipeline
            transformed_support_images.append(self.processing_transform(support_images[i]))
        support_images_tensor = torch.stack(transformed_support_images)

        transformed_query_images = []
        for i in range(query_images.shape[0]):
            transformed_query_images.append(self.processing_transform(query_images[i]))
        query_images_tensor = torch.stack(transformed_query_images)

        support_labels_tensor = torch.from_numpy(support_labels).long()
        query_labels_tensor = torch.from_numpy(query_labels).long() 

        # Apply user-specified transform if any, after default transforms (applied to torch tensors)
        if self.user_transform:
            # This assumes user_transform expects a batch of (C,H,W) tensors if applied to stack, or single if per image
            # For simplicity, let's assume it's per image if provided here, consistent with original logic.
            support_images_tensor = torch.stack([self.user_transform(img) for img in support_images_tensor])
            query_images_tensor = torch.stack([self.user_transform(img) for img in query_images_tensor])

        return support_images_tensor, support_labels_tensor, query_images_tensor, query_labels_tensor

    def close(self):
        if hasattr(self, '_file') and self._file:
            self._file.close()
            self._file = None

    def __del__(self):
        self.close()

# --- Argument Parsing (remains mostly the same) ---
def parse_args():
    parser = argparse.ArgumentParser(description="Meta-Training on Naturalistic Same-Different Dataset")

    # Paths
    parser.add_argument('--data_dir', type=str, default='/scratch/gpfs/mg7411/samedifferent/data/naturalistic',
                        help='Directory containing train.h5, val.h5, test.h5')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs and model checkpoints')

    # Model
    parser.add_argument('--model', type=str, required=True, choices=['conv2lr', 'conv4lr', 'conv6lr'],
                        help='Which Conv-N LR model architecture to use')

    # Meta-Learning Hyperparameters
    parser.add_argument('--epochs', type=int, default=60, help='Number of meta-training epochs')
    parser.add_argument('--episodes_per_epoch', type=int, default=1000, help='Number of episodes per training epoch')
    parser.add_argument('--meta_lr', type=float, default=1e-5, help='Meta-optimizer learning rate (outer loop)')
    parser.add_argument('--inner_lr', type=float, default=1e-5, help='Inner loop learning rate')
    parser.add_argument('--inner_steps', type=int, default=5, help='Number of adaptation steps in inner loop (from reference script)')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for the meta-optimizer (AdamW). Default: 0.0 (no decay)')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0, help='Max norm for gradient clipping in the outer loop. Default: 1.0')
    parser.add_argument('--first_order', action='store_true', help='Use first-order MAML instead of second-order.')

    # Training Settings
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader (0 often safest for HDF5)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--log_interval', type=int, default=50, help='Log training status every N episodes')
    parser.add_argument('--val_interval', type=int, default=1, help='Run validation every N epochs')
    parser.add_argument('--max_val_episodes', type=int, default=None,
                        help='Maximum number of episodes to use for validation. Default: all episodes in val set.')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('--min_delta', type=float, default=0.001, help='Minimum improvement for early stopping')

    return parser.parse_args()

# --- MAML Helper Function (Manual Implementation) ---
def fast_adapt(batch, learner, loss_fn, adaptation_steps, device, episode_idx_debug="N/A", force_first_order_adapt=False):
    """
    Manual implementation of MAML adaptation and evaluation step.
    Derived from typical learn2learn usage.
    Returns evaluation loss and accuracy on the query set.
    """
    support_images, support_labels, query_images, query_labels = batch
    support_images, support_labels = support_images.to(device), support_labels.to(device)
    query_images, query_labels = query_images.to(device), query_labels.to(device)

    # Store original training mode of the learner (MAML wrapper) and set to train for adaptation
    original_learner_training_state = learner.training
    learner.train() 

    # Determine if this adaptation step should be first_order
    is_adapt_first_order = learner.first_order # Default from learner
    if force_first_order_adapt: # If True, force it for this call (e.g., during validation)
        is_adapt_first_order = True

    # Adapt the model
    for step in range(adaptation_steps):
        # Enable gradient computation for this specific adaptation step's forward pass
        with torch.enable_grad():
            # --- Crucial fix for validation context ---
            if episode_idx_debug.startswith("val_ep") or force_first_order_adapt: # More general condition for safety
                # When in validation (or forced first_order), ensure learner params require grad after cloning
                for p in learner.module.parameters():
                    p.requires_grad_(True)
            # --- End crucial fix ---

            adaptation_logits = learner(support_images) # Calls learner.forward -> learner.module.forward
            adaptation_error = loss_fn(adaptation_logits, support_labels)

            if torch.isnan(adaptation_error) or torch.isinf(adaptation_error) or adaptation_error.item() > 1e6:
                print(f"WARNING (fast_adapt {episode_idx_debug}, Inner Step {step+1}): Unstable adaptation_error: {adaptation_error.item()}")

            # --- Critical Debug for Validation --- 
            # if episode_idx_debug.startswith("val_ep") and step == 0: # Only print for first validation step once per episode
            #     print(f"DEBUG VALIDATION (Episode: {episode_idx_debug}, Inner Step: {step+1}):")
            #     print(f"  adaptation_error: val={adaptation_error.item():.4f}, requires_grad={adaptation_error.requires_grad}, grad_fn is set: {adaptation_error.grad_fn is not None}")
            #     # Check a few learner parameters
            #     learner_params_req_grad = [(n, p.requires_grad) for n, p in learner.module.named_parameters()]
            #     print(f"  Learner param requires_grad status (first 3): {learner_params_req_grad[:3]}")
            #     if not all(p[1] for p in learner_params_req_grad):
            #         print(f"  WARNING: Some learner params do not require grad: {[n for n,rg in learner_params_req_grad if not rg]}")
            # --- End Critical Debug ---
            
            if adaptation_error.isnan():
                print(f"ERROR: NaN loss detected at Episode {episode_idx_debug}, Inner step {step+1}. Stopping adaptation.")
                # Optionally, you could raise an error here or return a specific indicator
                # For now, we'll let it proceed to learner.adapt which will likely raise the RuntimeError
                # This helps confirm the NaN is the source.
                # To prevent the RuntimeError and debug further, one might return (NaN, NaN) from fast_adapt here.

        # Compute gradients and update the learner (cloned model)
        # DEBUG PRINT TO CONFIRM (can be removed later)
        # print(f"DEBUG ADAPT CALL (Episode: {episode_idx_debug}, Step: {step+1}): learner.first_order={learner.first_order}, force_first_order_adapt={force_first_order_adapt}, is_adapt_first_order={is_adapt_first_order}, adaptation_error.requires_grad={adaptation_error.requires_grad}, adaptation_error.grad_fn={adaptation_error.grad_fn is not None}")
        learner.adapt(adaptation_error, first_order=is_adapt_first_order) # MODIFIED: Pass determined first_order status

    # Restore original training mode of the learner for the final evaluation on the query set
    learner.train(original_learner_training_state) 

    # Evaluate the adapted model on query data
    # This forward pass will use the original_learner_training_state (e.g., eval mode if called from validation)
    evaluation_logits = learner(query_images)
    evaluation_error = loss_fn(evaluation_logits, query_labels)
    evaluation_acc = accuracy(evaluation_logits, query_labels)

    if torch.isnan(evaluation_error) or torch.isinf(evaluation_error) or evaluation_error.item() > 1e6:
        print(f"WARNING (fast_adapt {episode_idx_debug}): Unstable evaluation_error: {evaluation_error.item()}")

    return evaluation_error, evaluation_acc

# --- Main Training Function ---
def main():
    args = parse_args()

    # Setup (includes seeding)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        # Optional: torch.backends.cudnn.deterministic = True
        # Optional: torch.backends.cudnn.benchmark = False

    device = torch.device(args.device)
    # Create log dir specific to model and seed
    log_dir = Path(args.log_dir) / args.model / f"seed_{args.seed}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Args: {args}")
    print(f"Using device: {device}")
    print(f"Logging to: {log_dir}")

    # --- Load Model --- Instantiation based on reference ---
    print(f"Loading model: {args.model}")
    try:
        model = ARCHITECTURES[args.model]() # Instantiate using the dictionary
    except KeyError:
        raise ValueError(f"Unknown model type: {args.model}")

    # --- Apply Initialization (from reference script) ---
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.01)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0.01)
    print("Applied Kaiming Normal init to Conv2d and constants to BatchNorm2d.")

    model.to(device)
    # Save initial model weights before wrapping with MAML or starting training
    initial_model_save_path = log_dir / "initial_model.pth"
    try:
        torch.save(model.state_dict(), initial_model_save_path)
        print(f"Saved initial model weights to {initial_model_save_path}")
    except Exception as e:
        print(f"Error saving initial model weights: {e}")
    # print(model) # Optional: Print model structure

    # --- Wrap model with MAML --- based on reference ---
    maml = l2l.algorithms.MAML(
        model,
        lr=args.inner_lr,
        first_order=args.first_order, # Use first_order flag from args
        allow_unused=True, # Allow unused parameters if model has them
        allow_nograd=True  # Allow no grad for params not used in inner loop
    )
    maml.to(device) # Ensure MAML wrapper and cloned model are on device

    # --- Load Data ---
    transform = None # No extra transforms for now
    try:
        train_dataset = MetaNaturalisticDataset(Path(args.data_dir) / 'train.h5', transform=transform)
        val_dataset = MetaNaturalisticDataset(Path(args.data_dir) / 'val.h5', transform=transform)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        # Ensure file handles are closed if datasets were partially initialized
        if 'train_dataset' in locals() and hasattr(train_dataset, 'close'): train_dataset.close()
        if 'val_dataset' in locals() and hasattr(val_dataset, 'close'): val_dataset.close()
        return

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # --- Setup Optimizer and Loss --- based on reference ---
    meta_optimizer = optim.AdamW(maml.parameters(), lr=args.meta_lr, weight_decay=args.weight_decay)
    # Assuming CrossEntropyLoss based on typical MAML classification and accuracy function
    criterion = nn.CrossEntropyLoss()
    # Mixed precision scaler (optional but good practice if using CUDA)
    scaler = torch.amp.GradScaler(device=args.device if args.device == 'cuda' else 'cpu', enabled=(args.device == 'cuda' and torch.cuda.is_available()))

    # --- Training Loop --- incorporating MAML steps ---
    print("Starting meta-training...")
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        maml.train() # Set MAML (and base model) to train mode
        meta_train_loss = 0.0
        meta_train_acc = 0.0
        
        pbar = tqdm(train_loader, total=args.episodes_per_epoch, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        
        for i, batch in enumerate(pbar):
            if i >= args.episodes_per_epoch:
                break
            
            meta_optimizer.zero_grad()
            
            # Squeeze the meta-batch dimension (size 1)
            # We handle the single episode directly
            task_batch = [d.squeeze(0) for d in batch] 

            # --- MAML Inner and Outer Loop --- 
            # Using Automatic Mixed Precision
            # Use torch.amp (newer API); device_type is only for cuda, autocast still runs on CPU but may have no effect for float32.
            # For CPU, enabled should ideally be False if we only want AMP for CUDA.
            is_cuda_and_amp_enabled = args.device == 'cuda' and torch.cuda.is_available()
            with torch.amp.autocast(device_type=args.device if is_cuda_and_amp_enabled else 'cpu', dtype=torch.bfloat16 if is_cuda_and_amp_enabled else torch.float32, enabled=is_cuda_and_amp_enabled):
                # Create a functional clone of the model for adaptation
                learner = maml.clone()
                
                # Adapt model and get query loss/acc for this task
                evaluation_loss, evaluation_acc = fast_adapt(task_batch, 
                                                               learner, 
                                                               criterion, 
                                                               args.inner_steps, 
                                                               device,
                                                               episode_idx_debug=f"train_ep_{i}",
                                                               force_first_order_adapt=False)
            
            # Average the evaluation loss/acc across tasks in the meta-batch (batch_size=1 here)
            # For meta-batch > 1, you would average evaluation_loss across the batch before backward
            meta_batch_loss = evaluation_loss 
            meta_batch_acc = evaluation_acc

            # Backpropagate meta-loss and update MAML parameters
            if is_cuda_and_amp_enabled:
                scaler.scale(meta_batch_loss).backward()
                scaler.unscale_(meta_optimizer) # Unscale before clipping
                # Calculate and print grad norm before clipping
                total_norm_before_clip = 0.0
                for p in maml.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm_before_clip += param_norm.item() ** 2
                total_norm_before_clip = total_norm_before_clip ** 0.5
                # if i % args.log_interval == 0 or total_norm_before_clip > args.grad_clip_norm * 1.5 : # Log periodically or if norm is high
                #     print(f"DEBUG (Epoch {epoch}, Episode {i}): Grad norm before clip: {total_norm_before_clip:.4f} (Clip at: {args.grad_clip_norm})")
                torch.nn.utils.clip_grad_norm_(maml.parameters(), max_norm=args.grad_clip_norm)
                scaler.step(meta_optimizer)
                scaler.update()
            else: # CPU or AMP disabled
                meta_batch_loss.backward()
                # Calculate and print grad norm before clipping
                total_norm_before_clip = 0.0
                for p in maml.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm_before_clip += param_norm.item() ** 2
                total_norm_before_clip = total_norm_before_clip ** 0.5
                # if i % args.log_interval == 0 or total_norm_before_clip > args.grad_clip_norm * 1.5: # Log periodically or if norm is high
                #      print(f"DEBUG (Epoch {epoch}, Episode {i}): Grad norm before clip: {total_norm_before_clip:.4f} (Clip at: {args.grad_clip_norm})")
                torch.nn.utils.clip_grad_norm_(maml.parameters(), max_norm=args.grad_clip_norm)
                meta_optimizer.step()

            meta_train_loss += meta_batch_loss.item()
            meta_train_acc += meta_batch_acc.item()

            pbar.set_postfix(loss=f"{meta_batch_loss.item():.4f}", acc=f"{meta_batch_acc.item():.3f}")
            
            # Free memory
            del learner, task_batch, evaluation_loss, evaluation_acc
            if device == 'cuda': torch.cuda.empty_cache() 

        avg_epoch_loss = meta_train_loss / args.episodes_per_epoch
        avg_epoch_acc = meta_train_acc / args.episodes_per_epoch
        print(f"Epoch {epoch}/{args.epochs} [Train] Avg Loss: {avg_epoch_loss:.4f}, Avg Acc: {avg_epoch_acc:.3f}")
        gc.collect()

        # --- Validation Loop --- (similar structure, no grad and no meta-update)
        if epoch % args.val_interval == 0:
            maml.eval() # Set MAML (and base model) to eval mode
            meta_val_loss = 0.0
            meta_val_acc = 0.0
            val_episodes = 0

            # Determine the number of validation episodes to run
            num_val_episodes_to_run = len(val_loader)
            if args.max_val_episodes is not None and args.max_val_episodes > 0:
                num_val_episodes_to_run = min(len(val_loader), args.max_val_episodes)
                print(f"Running validation on a subset of {num_val_episodes_to_run} episodes (out of {len(val_loader)} total).")

            pbar_val = tqdm(val_loader, total=num_val_episodes_to_run, desc=f"Epoch {epoch}/{args.epochs} [Val]")
            
            with torch.no_grad():
                for i, batch in enumerate(pbar_val):
                    if i >= num_val_episodes_to_run:
                        break # Stop after processing the desired number of validation episodes

                    task_batch = [d.squeeze(0) for d in batch]
                    
                    # Using Automatic Mixed Precision (optional for eval)
                    # For eval, autocast can still be used for consistency, but grads are not an issue.
                    with torch.amp.autocast(device_type=args.device if is_cuda_and_amp_enabled else 'cpu', dtype=torch.bfloat16 if is_cuda_and_amp_enabled else torch.float32, enabled=is_cuda_and_amp_enabled):
                        learner = maml.clone() # Clone for adaptation, even in eval
                        evaluation_loss, evaluation_acc = fast_adapt(task_batch, 
                                                                       learner, 
                                                                       criterion, 
                                                                       args.inner_steps, # Use train steps or define separate val_steps? 
                                                                       device,
                                                                       episode_idx_debug=f"val_ep_{val_episodes}",
                                                                       force_first_order_adapt=True) # MODIFIED: Force first_order for validation
                    
                    meta_val_loss += evaluation_loss.item()
                    meta_val_acc += evaluation_acc.item()
                    val_episodes += 1
                    pbar_val.set_postfix(loss=f"{evaluation_loss.item():.4f}", acc=f"{evaluation_acc.item():.3f}")
                    
                    del learner, task_batch, evaluation_loss, evaluation_acc
                    if device == 'cuda': torch.cuda.empty_cache()

            avg_val_loss = meta_val_loss / val_episodes if val_episodes > 0 else 0
            avg_val_acc = meta_val_acc / val_episodes if val_episodes > 0 else 0
            print(f"Epoch {epoch}/{args.epochs} [Val]   Avg Loss: {avg_val_loss:.4f}, Avg Acc: {avg_val_acc:.3f}")
            gc.collect()

            # --- Checkpoint Saving & Early Stopping --- based on reference ---
            if avg_val_acc - best_val_acc > args.min_delta:
                best_val_acc = avg_val_acc
                patience_counter = 0
                save_path = log_dir / f"{args.model}_best.pth"
                try:
                     torch.save({
                         'epoch': epoch,
                         'model_state_dict': model.state_dict(), # Save base model state
                         'maml_state_dict': maml.state_dict(),   # Save MAML wrapper state
                         'optimizer_state_dict': meta_optimizer.state_dict(),
                         'best_val_acc': best_val_acc,
                         'args': args
                     }, save_path)
                     print(f"New best validation accuracy: {best_val_acc:.3f}. Saving model to {save_path}")
                except Exception as e:
                     print(f"Error saving checkpoint: {e}")
            else:
                patience_counter += 1
                print(f"Validation accuracy did not improve enough. Patience: {patience_counter}/{args.patience}")

            if args.checkpoint_interval > 0 and epoch % args.checkpoint_interval == 0:
                 save_path = log_dir / f"{args.model}_epoch_{epoch}.pth"
                 try:
                      torch.save({
                          'epoch': epoch,
                          'model_state_dict': model.state_dict(),
                          'maml_state_dict': maml.state_dict(),
                          'optimizer_state_dict': meta_optimizer.state_dict(),
                          'val_acc': avg_val_acc,
                          'args': args
                      }, save_path)
                      print(f"Saving checkpoint to {save_path}")
                 except Exception as e:
                     print(f"Error saving checkpoint: {e}")
            
            if patience_counter >= args.patience:
                 print(f"Early stopping triggered after {epoch} epochs.")
                 break # Exit epoch loop

    print("Meta-training finished.")
    print(f"Best validation accuracy: {best_val_acc:.3f}")

    # --- Cleanup ---
    train_dataset.close()
    val_dataset.close()
    # Consider adding test loop here if needed, loading best model

    # --- Test Loop ---
    print("\nStarting testing phase...")
    test_h5_path = Path(args.data_dir) / 'test.h5'
    if not test_h5_path.exists():
        print(f"Test data file not found: {test_h5_path}. Skipping testing.")
    else:
        try:
            test_dataset = MetaNaturalisticDataset(test_h5_path, transform=transform)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
        except Exception as e:
            print(f"Error loading test dataset: {e}. Skipping testing.")
            if 'test_dataset' in locals() and hasattr(test_dataset, 'close'): test_dataset.close()
            test_loader = None

        if test_loader:
            best_model_path = log_dir / f"{args.model}_best.pth"
            if best_model_path.exists():
                print(f"Loading best model for testing from: {best_model_path}")
                try:
                    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
                    # Load base model state
                    model.load_state_dict(checkpoint['model_state_dict'])
                    # Load MAML wrapper state (includes parameters of the meta-optimizer if maml itself has params, but mainly for first_order, lr etc.)
                    # The actual model parameters are within 'model'. MAML uses these.
                    # If MAML was saved with its own state_dict (e.g. for learned LR in some variants, or specific MAML params)
                    if 'maml_state_dict' in checkpoint:
                         maml.load_state_dict(checkpoint['maml_state_dict'])
                    else: # Fallback for older checkpoints or if maml_state_dict isn't strictly needed for inference pass
                         # Re-initialize MAML with the loaded model if only model_state_dict was present
                         # This ensures the MAML wrapper has the correct, trained model parameters.
                         maml = l2l.algorithms.MAML(
                             model, # The model now has the loaded best weights
                             lr=args.inner_lr,
                             first_order=False, # As used in training
                             allow_unused=True,
                             allow_nograd=True
                         )
                         maml.to(device)
                    print("Best model loaded successfully.")
                except Exception as e:
                    print(f"Error loading best model checkpoint: {e}. Skipping testing.")
                    test_loader = None # Prevent testing if model load fails

            else:
                print(f"Best model checkpoint not found at {best_model_path}. Skipping testing.")
                test_loader = None

            if test_loader:
                maml.eval() # Set MAML (and base model) to eval mode
                meta_test_loss = 0.0
                meta_test_acc = 0.0
                test_episodes = 0
                
                is_cuda_and_amp_enabled = args.device == 'cuda' and torch.cuda.is_available() # Re-check based on args for safety

                pbar_test = tqdm(test_loader, desc="Testing")
                
                # No gradients needed for testing
                with torch.no_grad():
                    for batch_idx, batch in enumerate(pbar_test):
                        task_batch = [d.squeeze(0) for d in batch]
                        
                        with torch.amp.autocast(device_type=args.device if is_cuda_and_amp_enabled else 'cpu', dtype=torch.bfloat16 if is_cuda_and_amp_enabled else torch.float32, enabled=is_cuda_and_amp_enabled):
                            learner = maml.clone() # Clone for adaptation
                            evaluation_loss, evaluation_acc = fast_adapt(task_batch,
                                                                           learner,
                                                                           criterion,
                                                                           args.inner_steps, # Adaptation steps
                                                                           device,
                                                                           episode_idx_debug=f"test_ep_{batch_idx}",
                                                                           force_first_order_adapt=True) # Force first_order for test
                        
                        meta_test_loss += evaluation_loss.item()
                        meta_test_acc += evaluation_acc.item()
                        test_episodes += 1
                        pbar_test.set_postfix(loss=f"{evaluation_loss.item():.4f}", acc=f"{evaluation_acc.item():.3f}")

                        del learner, task_batch, evaluation_loss, evaluation_acc
                        if device == 'cuda': torch.cuda.empty_cache()

                if test_episodes > 0:
                    avg_test_loss = meta_test_loss / test_episodes
                    avg_test_acc = meta_test_acc / test_episodes
                    print(f"Test Results: Avg Loss: {avg_test_loss:.4f}, Avg Acc: {avg_test_acc:.3f} across {test_episodes} episodes.")
                else:
                    print("No episodes processed in the test set.")
                
                gc.collect()
            
            test_dataset.close()

if __name__ == "__main__":
    main() 
