#!/bin/bash
#SBATCH --job-name=comp_efficiency
#SBATCH --output=results/computational_efficiency/slurm_output_%j.out
#SBATCH --error=results/computational_efficiency/slurm_error_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mg7411@princeton.edu

echo "=================================================="
echo "COMPUTATIONAL EFFICIENCY COMPARISON EXPERIMENT"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "=================================================="

# Environment setup
module load anaconda3
module load cuda/11.7.0
conda activate tensorflow

# Navigate to project directory
cd /scratch/gpfs/mg7411/samedifferent

# Create output directory
mkdir -p results/computational_efficiency

# Print environment info
echo "Python version: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"

echo ""
echo "Starting computational efficiency comparison..."
echo "Testing hypothesis: FOMAML (fast training, slow adaptation) vs Second-Order MAML (slow training, fast adaptation)"
echo ""

# Run the computational efficiency comparison
python scripts/computational_efficiency_comparison.py \
    --data_dir /scratch/gpfs/mg7411/samedifferent/data/meta_h5/pb \
    --save_dir results/computational_efficiency \
    --meta_batch_size 8 \
    --inner_lr 0.001 \
    --outer_lr 0.0001 \
    --max_training_batches 300 \
    --warmup_batches 800 \
    --training_adaptation_steps 5 \
    --max_test_episodes 150 \
    --max_adaptation_steps 20 \
    --target_accuracies 55.0 60.0 65.0 70.0 \
    --device cuda

echo ""
echo "=================================================="
echo "Job completed at: $(date)"
echo "=================================================="

# Print final summary
echo ""
echo "📊 Results summary:"
if [ -f "results/computational_efficiency/efficiency_summary.txt" ]; then
    echo "✅ Summary file created successfully"
    echo "Key findings:"
    head -20 results/computational_efficiency/efficiency_summary.txt
else
    echo "❌ Summary file not found - check for errors"
fi

echo ""
echo "📁 Output files:"
ls -la results/computational_efficiency/

echo ""
echo "🎉 Computational efficiency experiment completed!"
echo "Check results/computational_efficiency/ for detailed results and plots" 