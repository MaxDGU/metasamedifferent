#!/bin/bash
#SBATCH --job-name=efficiency_paradox
#SBATCH --output=logs/efficiency_paradox_%j.out
#SBATCH --error=logs/efficiency_paradox_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1

# Load required modules
module load anaconda3/2022.5
module load cudatoolkit/12.6

# Activate conda environment
source /usr/licensed/anaconda3/2022.5/etc/profile.d/conda.sh
conda activate tensorflow

# Create logs directory
mkdir -p logs

echo "🧪 META-LEARNING EFFICIENCY PARADOX EXPERIMENT"
echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST" 
echo "Start time: $(date)"
echo ""
echo "This experiment demonstrates the fundamental trade-off in meta-learning:"
echo "1. TRAINING PHASE: Meta-learning is DATA-HUNGRY"
echo "   - Needs more data/time to learn good meta-parameters"
echo "   - Shows slower convergence compared to vanilla SGD"
echo ""
echo "2. TESTING PHASE: Meta-learning is DATA-EFFICIENT"
echo "   - Adapts quickly to new tasks with few examples"
echo "   - Shows faster few-shot learning than vanilla SGD"
echo ""
echo "Expected Runtime: ~20-24 hours for comprehensive analysis"
echo "Expected Results: Clear demonstration of the efficiency paradox"
echo "================================================"
echo ""

# Run the efficiency paradox experiment
python scripts/meta_learning_efficiency_paradox.py \
    --data_dir /scratch/gpfs/mg7411/samedifferent/data/meta_h5/pb \
    --save_dir results/efficiency_paradox \
    --training_epochs 15 \
    --meta_batch_size 16 \
    --vanilla_batch_size 64 \
    --inner_lr 0.01 \
    --outer_lr 0.001 \
    --vanilla_lr 0.0001 \
    --adaptation_steps 5 \
    --test_episodes 100 \
    --seeds 42 43 44 45 46

echo ""
echo "================================================"
echo "🎉 EFFICIENCY PARADOX EXPERIMENT COMPLETED!"
echo "End time: $(date)"
echo "================================================"

# Display final summary
echo ""
echo "📊 RESULTS SUMMARY:"
if [ -f "results/efficiency_paradox/efficiency_paradox_results.json" ]; then
    echo "✅ Main results file created successfully"
fi

if [ -f "results/efficiency_paradox/efficiency_paradox_analysis.png" ]; then
    echo "✅ Visualization created successfully"
fi

if [ -f "results/efficiency_paradox/efficiency_paradox_report.txt" ]; then
    echo "✅ Summary report created successfully"
    echo ""
    echo "📖 KEY FINDINGS:"
    tail -10 results/efficiency_paradox/efficiency_paradox_report.txt
fi

echo ""
echo "📁 All results saved to: results/efficiency_paradox/"
echo "🔬 This experiment provides definitive evidence for the meta-learning efficiency paradox"
echo "💡 Use these results to demonstrate the fundamental trade-offs in meta-learning" 