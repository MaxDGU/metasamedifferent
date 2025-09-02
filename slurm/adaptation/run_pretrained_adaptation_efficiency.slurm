#!/bin/bash
#SBATCH --job-name=pretrained_adaptation
#SBATCH --output=results/pretrained_adaptation_efficiency/slurm_output_%j.out
#SBATCH --error=results/pretrained_adaptation_efficiency/slurm_error_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mg7411@princeton.edu

echo "=================================================="
echo "PRETRAINED MODEL ADAPTATION EFFICIENCY EXPERIMENT"
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
mkdir -p results/pretrained_adaptation_efficiency

# Print environment info
echo "Python version: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"

echo ""
echo "Starting pretrained model adaptation efficiency test..."
echo "Testing hypothesis: Second-Order MAML adapts faster than First-Order MAML"
echo "Using pretrained models from: /scratch/gpfs/mg7411/samedifferent/results/meta_baselines/conv6/"
echo ""

# Run the pretrained adaptation efficiency test
python scripts/test_adaptation_efficiency_pretrained.py \
    --data_dir /scratch/gpfs/mg7411/samedifferent/data/meta_h5/pb \
    --model_base_path /scratch/gpfs/mg7411/samedifferent/results/meta_baselines/conv6 \
    --save_dir results/pretrained_adaptation_efficiency \
    --batch_size 8 \
    --inner_lr 0.01 \
    --max_test_episodes 100 \
    --max_adaptation_steps 20 \
    --target_accuracies 55.0 60.0 65.0 70.0

echo ""
echo "=================================================="
echo "Job completed at: $(date)"
echo "=================================================="

# Print final summary
echo ""
echo "📊 Results summary:"
if [ -f "results/pretrained_adaptation_efficiency/adaptation_summary.txt" ]; then
    echo "✅ Summary file created successfully"
    echo "Key findings:"
    head -20 results/pretrained_adaptation_efficiency/adaptation_summary.txt
else
    echo "❌ Summary file not found - check for errors"
fi

echo ""
echo "📁 Output files:"
ls -la results/pretrained_adaptation_efficiency/

echo ""
echo "🎉 Pretrained adaptation efficiency experiment completed!"
echo "Check results/pretrained_adaptation_efficiency/ for detailed results and plots" 