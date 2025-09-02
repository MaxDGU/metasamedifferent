#!/bin/bash
#SBATCH --job-name=sample_efficiency_full
#SBATCH --output=logs/sample_efficiency_full_%j.out
#SBATCH --error=logs/sample_efficiency_full_%j.err
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1

# Load required modules
module load anaconda3/2022.5
module load cudatoolkit/12.6

# Activate conda environment (adjust path as needed)

#source activate /home/mg7411/.conda/envs/metasamedifferent
source /usr/licensed/anaconda3/2022.5/etc/profile.d/conda.sh

conda activate tensorflow
# Set working directory
#cd /scratch/gpfs/mg7411/samedifferent/metasamedifferent

# Create logs directory
mkdir -p logs

# Run the sample efficiency comparison with optimized parameters for speed
echo "Starting FULL sample efficiency comparison with optimized parameters..."
echo "Key optimizations:"
echo "- Reduced epochs: 50 → 15"
echo "- Increased meta_batch_size: 8 → 16 (fewer batches per epoch)"
echo "- Increased vanilla_batch_size: 32 → 64 (fewer batches per epoch)"
echo "- Reduced adaptation_steps: 5 → 3 (40% faster adaptation)"
echo "- Reduced validation frequency: 500 → 3000 (66% fewer validations)"
echo "- Running all three methods: FOMAML + Second-Order MAML + Vanilla SGD"
echo ""

python scripts/sample_efficiency_comparison.py \
    --epochs 15 \
    --seed 42 \
    --meta_batch_size 16 \
    --vanilla_batch_size 64 \
    --inner_lr 0.05 \
    --outer_lr 0.001 \
    --vanilla_lr 0.0001 \
    --adaptation_steps 3 \
    --val_frequency 3000 \
    --save_dir results/sample_efficiency_full \
    --methods fomaml second_order vanilla \
    --data_dir /scratch/gpfs/mg7411/samedifferent/data/meta_h5/pb

# After completion, generate plots
python scripts/plot_sample_efficiency_results.py \
    --results_dir results/sample_efficiency_full/seed_42

echo "Full sample efficiency comparison completed!"
echo "Results saved to: results/sample_efficiency_full/seed_42/" 