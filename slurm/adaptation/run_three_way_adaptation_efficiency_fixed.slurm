#!/bin/bash
#SBATCH --job-name=three_way_fixed
#SBATCH --output=results/three_way_adaptation_efficiency_fixed/slurm_output_%j.out
#SBATCH --error=results/three_way_adaptation_efficiency_fixed/slurm_error_%j.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mg7411@princeton.edu

echo "=================================================="
echo "THREE-WAY ADAPTATION EFFICIENCY COMPARISON (FIXED)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "=================================================="

# Environment setup
module load anaconda3 2>/dev/null || echo "Note: anaconda3 module not found, using system Python"
module load cuda/11.7.0 2>/dev/null || echo "Note: CUDA module not found, using system CUDA"
conda activate tensorflow

# Navigate to project directory
cd /scratch/gpfs/mg7411/samedifferent

# Create output directory
mkdir -p results/three_way_adaptation_efficiency_fixed

# Print environment info
echo "Python version: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"

echo ""
echo "🔧 FIXED: Testing with EQUAL learning rates and fair comparison"
echo "Testing adaptation speed on naturalistic data:"
echo "  • First-Order MAML (LR: 0.001)"
echo "  • Second-Order MAML (LR: 0.001)"  
echo "  • Vanilla SGD (LR: 0.001)"
echo "All methods now use the same learning rate for fair comparison!"
echo ""

# Run the three-way adaptation efficiency comparison with FIXED parameters
python scripts/test_adaptation_efficiency_three_way.py \
    --naturalistic_data_path /scratch/gpfs/mg7411/samedifferent/data/naturalistic/test.h5 \
    --meta_model_base_path /scratch/gpfs/mg7411/samedifferent/results/meta_baselines/conv6 \
    --vanilla_model_base_path /scratch/gpfs/mg7411/samedifferent/results/pb_baselines_vanilla_final/all_tasks/conv6/test_regular \
    --save_dir results/three_way_adaptation_efficiency_fixed \
    --inner_lr 0.001 \
    --vanilla_lr 0.001 \
    --max_test_episodes 100 \
    --max_adaptation_steps 30 \
    --target_accuracies 55.0 60.0 65.0 70.0

echo ""
echo "=================================================="
echo "Job completed at: $(date)"
echo "=================================================="

# Print final summary
echo ""
echo "📊 Results summary:"
if [ -f "results/three_way_adaptation_efficiency_fixed/three_way_adaptation_results.json" ]; then
    echo "✅ Results file created successfully"
    echo "Key findings from the FIXED comparison:"
    python -c "
import json
try:
    with open('results/three_way_adaptation_efficiency_fixed/three_way_adaptation_results.json', 'r') as f:
        results = json.load(f)
    for method in ['fomaml', 'second_order', 'vanilla_sgd']:
        if method in results and results[method]['curves']:
            curves = results[method]['curves']
            final_accs = [curve[-1] for curve in curves]
            import numpy as np
            method_name = {'fomaml': 'First-Order MAML', 'second_order': 'Second-Order MAML', 'vanilla_sgd': 'Vanilla SGD'}[method]
            print(f'{method_name}: {np.mean(final_accs):.1f}% ± {np.std(final_accs):.1f}%')
except:
    print('Could not parse results file')
"
else
    echo "❌ Results file not found - check for errors"
fi

echo ""
echo "📁 Output files:"
ls -la results/three_way_adaptation_efficiency_fixed/

echo ""
echo "🎉 FIXED three-way adaptation efficiency experiment completed!"
echo "Check results/three_way_adaptation_efficiency_fixed/ for corrected results"

echo ""
echo "📋 COMPARISON NOTES:"
echo "This experiment fixes the learning rate mismatch:"
echo "  - Original: Meta (0.01) vs Vanilla (0.001) - 10x difference!"
echo "  - Fixed: Both use 0.001 for fair comparison"
echo "  - Meta-learning should now show expected advantage" 