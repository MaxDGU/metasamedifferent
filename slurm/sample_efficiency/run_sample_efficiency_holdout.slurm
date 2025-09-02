#!/bin/bash
#SBATCH --job-name=sample_efficiency_ood
#SBATCH --output=logs/sample_efficiency_ood_%j.out
#SBATCH --error=logs/sample_efficiency_ood_%j.err
#SBATCH --time=16:00:00
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

# Run the OOD sample efficiency comparison with holdout task
echo "Starting OUT-OF-DISTRIBUTION Sample Efficiency Comparison..."
echo "=========================================================="
echo "This experiment tests meta-learning's TRUE advantage:"
echo "- Training on 9 tasks, validating on 1 HELD-OUT task"
echo "- Tests novel task generalization (OOD performance)"
echo "- Should show meta-learning superiority over vanilla SGD"
echo ""
echo "Holdout task: 'scrambled' (visually challenging variant)"
echo "Training tasks: regular, lines, open, wider_line, random_color, arrows, irregular, filled, original"
echo ""
echo "Optimized parameters for efficiency:"
echo "- Epochs: 15 (focused evaluation)"
echo "- Meta batch size: 16 (balanced efficiency/performance)"
echo "- Vanilla batch size: 64 (efficient processing)"
echo "- Adaptation steps: 3 (sufficient for adaptation)"
echo "- Validation frequency: 3000 (reduced overhead)"
echo ""

python scripts/sample_efficiency_comparison_holdout.py \
    --epochs 15 \
    --seed 42 \
    --holdout_task scrambled \
    --meta_batch_size 16 \
    --vanilla_batch_size 64 \
    --inner_lr 0.05 \
    --outer_lr 0.001 \
    --vanilla_lr 0.0001 \
    --adaptation_steps 3 \
    --val_frequency 3000 \
    --save_dir results/sample_efficiency_ood \
    --methods fomaml second_order vanilla \
    --data_dir /scratch/gpfs/mg7411/samedifferent/data/meta_h5/pb

echo ""
echo "=========================================================="
echo "OOD Sample Efficiency Experiment Completed!"
echo "Results saved to: results/sample_efficiency_ood/holdout_scrambled/seed_42/"
echo ""
echo "Expected outcome: Meta-learning should outperform vanilla SGD"
echo "on the held-out 'scrambled' task due to better generalization."
echo "==========================================================" 