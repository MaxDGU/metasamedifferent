#!/bin/bash
#SBATCH --job-name=sample_efficiency_ood_long
#SBATCH --output=logs/sample_efficiency_ood_long_%j.out
#SBATCH --error=logs/sample_efficiency_ood_long_%j.err
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1

# Load required modules
module load anaconda3/2022.5
module load cudatoolkit/12.6

# Activate conda environment
source /usr/licensed/anaconda3/2022.5/etc/profile.d/conda.sh
conda activate tensorflow

# Create logs directory
mkdir -p logs

# Run the LONG OOD sample efficiency comparison with holdout task
echo "Starting OUT-OF-DISTRIBUTION Sample Efficiency Comparison (LONG VERSION)"
echo "======================================================================="
echo "This experiment tests meta-learning's TRUE advantage with thorough evaluation:"
echo "- Training on 9 tasks, validating on 1 HELD-OUT task"
echo "- Tests novel task generalization (OOD performance)"
echo "- Should show meta-learning superiority over vanilla SGD"
echo ""
echo "LONG VERSION PARAMETERS:"
echo "- Adaptation steps: 10 (training) / 15 (testing)"
echo "- Epochs: 25 (thorough evaluation)"
echo "- Validation frequency: 1000 (more frequent monitoring)"
echo "- Meta batch size: 16"
echo "- Expected runtime: 24-36 hours"
echo ""
echo "Holdout task: 'scrambled' (visually challenging variant)"
echo "Training tasks: regular, lines, open, wider_line, random_color, arrows, irregular, filled, original"
echo ""

python scripts/sample_efficiency_comparison_holdout_long.py \
    --epochs 25 \
    --seed 42 \
    --holdout_task scrambled \
    --meta_batch_size 16 \
    --vanilla_batch_size 64 \
    --inner_lr 0.05 \
    --outer_lr 0.001 \
    --vanilla_lr 0.0001 \
    --adaptation_steps 10 \
    --test_adaptation_steps 15 \
    --val_frequency 1000 \
    --save_dir results/sample_efficiency_ood_long \
    --methods fomaml second_order vanilla \
    --data_dir /scratch/gpfs/mg7411/samedifferent/data/meta_h5/pb

echo ""
echo "======================================================================="
echo "LONG OOD Sample Efficiency Experiment Completed!"
echo "Results saved to: results/sample_efficiency_ood_long/holdout_scrambled/seed_42/"
echo ""
echo "This thorough evaluation should demonstrate meta-learning's advantage"
echo "on novel task generalization with proper adaptation steps."
echo "=======================================================================" 