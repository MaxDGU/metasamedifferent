#!/bin/bash
#SBATCH --job-name=check-model-params
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:05:00
#SBATCH --output=logs/check_params_%A.out

# --- Environment Setup ---
source /usr/share/Modules/init/bash
module load anaconda3/2023.9
conda activate tensorflow

# --- Project Paths ---
export GIT_REPO_PATH="/scratch/gpfs/mg7411/samedifferent"
export PYTHONPATH="${PYTHONPATH}:${GIT_REPO_PATH}"
cd "${GIT_REPO_PATH}"

# --- Data and Model Paths on Della ---
META_NAT_MODEL="/scratch/gpfs/mg7411/samedifferent/logs_naturalistic_meta/conv6lr/seed_111/conv6lr/seed_111/conv6lr_best.pth"
VANILLA_NAT_MODEL="/scratch/gpfs/mg7411/samedifferent/logs_naturalistic_vanilla/conv6lr/seed_123/best_model.pt"
META_PB_MODEL="/scratch/gpfs/mg7411/samedifferent/results/pb_retrained_conv6lr/conv6/seed_789/best_model.pt"
VANILLA_PB_MODEL="/scratch/gpfs/mg7411/results/pb_baselines/regular/conv6/seed_50/best_model.pt"

# --- Execution ---
echo "Running model parameter check script..."
python -u scripts/check_model_parameters.py \
    --meta_pb_model "${META_PB_MODEL}" \
    --vanilla_pb_model "${VANILLA_PB_MODEL}" \
    --meta_nat_model "${META_NAT_MODEL}" \
    --vanilla_nat_model "${VANILLA_NAT_MODEL}" 