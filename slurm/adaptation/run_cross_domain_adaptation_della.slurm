#!/bin/bash
#SBATCH --job-name=cross-domain-pca
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/cross_domain_pca_%A.out

# --- Environment Setup ---
echo "Setting up the environment..."
source /usr/share/Modules/init/bash
module load anaconda3/2023.9
conda activate tensorflow
echo "Conda environment activated."

# --- Project Paths ---
export GIT_REPO_PATH="/scratch/gpfs/mg7411/samedifferent"
export PYTHONPATH="${PYTHONPATH}:${GIT_REPO_PATH}"
cd "${GIT_REPO_PATH}"
echo "Current working directory: $(pwd)"

# --- Data and Model Paths on Della ---
PB_DATA_DIR="/scratch/gpfs/mg7411/samedifferent/data/meta_h5/pb"
NAT_DATA_DIR="/scratch/gpfs/mg7411/samedifferent/data/naturalistic"
OUTPUT_DIR="${GIT_REPO_PATH}/visualizations/cross_domain_pca_final"

META_NAT_MODEL="/scratch/gpfs/mg7411/samedifferent/logs_naturalistic_meta/conv6lr/seed_111/conv6lr/seed_111/conv6lr_best.pth"
VANILLA_NAT_MODEL="/scratch/gpfs/mg7411/samedifferent/logs_naturalistic_vanilla/conv6lr/seed_123/best_model.pt"
META_PB_MODEL="/scratch/gpfs/mg7411/samedifferent/results/pb_retrained_conv6lr/conv6/seed_789/best_model.pt"
VANILLA_PB_MODEL="/scratch/gpfs/mg7411/samedifferent/results/pb_baselines_retrained/all_tasks/conv6/test_regular/seed_42/best_model.pt"


# --- Execution ---
echo "Starting Cross-Domain Adaptation PCA script..."
python -u scripts/visualize_cross_domain_adaptation.py \
    --pb_data_dir "${PB_DATA_DIR}" \
    --naturalistic_data_dir "${NAT_DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --meta_pb_model "${META_PB_MODEL}" \
    --vanilla_pb_model "${VANILLA_PB_MODEL}" \
    --meta_nat_model "${META_NAT_MODEL}" \
    --vanilla_nat_model "${VANILLA_NAT_MODEL}" \
    --shots 10 \
    --adaptation_steps 5 \
    --inner_lr 0.1

echo "Job finished with exit code $? at $(date)" 