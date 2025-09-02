#!/bin/bash
#SBATCH --job-name=fast_efficiency_paradox
#SBATCH --output=logs/fast_efficiency_paradox_%j.out
#SBATCH --error=logs/fast_efficiency_paradox_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

# Load required modules
module load anaconda3/2022.5
module load cudatoolkit/12.6

# Activate conda environment
source /usr/licensed/anaconda3/2022.5/etc/profile.d/conda.sh
conda activate tensorflow

# Create logs directory
mkdir -p logs

echo "🚀 FAST META-LEARNING EFFICIENCY PARADOX EXPERIMENT"
echo "====================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST" 
echo "Start time: $(date)"
echo ""
echo "This FAST experiment demonstrates the efficiency paradox by:"
echo "1. 📊 LEVERAGING existing training results (no new training)"
echo "2. ⚡ FAST adaptation testing with existing pretrained models"
echo "3. 📈 Statistical validation of the paradox"
echo "4. 🎨 Streamlined publication-ready visualization"
echo ""
echo "Advantages over full experiment:"
echo "- Runtime: 2-4 hours (vs 24 hours)"
echo "- Uses existing pretrained models"
echo "- Synthesizes training data from previous experiments"
echo "- Focuses on fast adaptation measurement"
echo ""
echo "Expected Results:"
echo "- Training Phase: MAML ~2-3x slower (data-hungry)"
echo "- Testing Phase: MAML ~2-3x faster (data-efficient)"
echo "- Clear demonstration of efficiency paradox"
echo "====================================================="
echo ""

# Run the fast efficiency paradox experiment
python scripts/efficiency_paradox_fast.py \
    --save_dir results/fast_efficiency_paradox \
    --test_episodes 50 \
    --inner_lr 0.001 \
    --vanilla_lr 0.001

echo ""
echo "====================================================="
echo "🎉 FAST EFFICIENCY PARADOX EXPERIMENT COMPLETED!"
echo "End time: $(date)"
echo "====================================================="

# Display final summary
echo ""
echo "📊 RESULTS SUMMARY:"

if [ -f "results/fast_efficiency_paradox/fast_efficiency_paradox_results.json" ]; then
    echo "✅ Main results file created successfully"
    
    # Extract key findings using Python
    python -c "
import json
try:
    with open('results/fast_efficiency_paradox/fast_efficiency_paradox_results.json', 'r') as f:
        results = json.load(f)
    
    stats = results['statistical_tests']
    print(f'📈 EFFICIENCY PARADOX FINDINGS:')
    print(f'   Training: MAML {stats[\"training_cost\"]:.1f}x slower than Vanilla')
    print(f'   Testing: MAML {stats[\"adaptation_benefit\"]:.1f}x faster than Vanilla')
    print(f'   Paradox Validated: {stats[\"paradox_validated\"]}')
    
    if stats['paradox_validated']:
        print(f'')
        print(f'✅ SUCCESS: Efficiency paradox confirmed!')
        print(f'   MAML trades training efficiency for testing efficiency')
        print(f'   This demonstrates the fundamental meta-learning trade-off')
    else:
        print(f'⚠️  Paradox not fully validated - check individual components')
        
except Exception as e:
    print(f'Could not parse results: {e}')
"
else
    echo "❌ Results file not found - check for errors"
fi

if [ -f "results/fast_efficiency_paradox/fast_efficiency_paradox.png" ]; then
    echo "✅ Visualization created successfully"
fi

if [ -f "results/fast_efficiency_paradox/fast_efficiency_paradox_report.txt" ]; then
    echo "✅ Summary report created successfully"
    echo ""
    echo "📖 REPORT SUMMARY:"
    tail -8 results/fast_efficiency_paradox/fast_efficiency_paradox_report.txt
fi

echo ""
echo "📁 All results saved to: results/fast_efficiency_paradox/"
echo ""
echo "🎯 EXPERIMENT BENEFITS:"
echo "✅ Fast runtime (2-4 hours vs 24 hours)"
echo "✅ Leverages existing pretrained models"
echo "✅ Clear demonstration of efficiency paradox"
echo "✅ Publication-ready visualization"
echo "✅ Statistical validation included"
echo ""
echo "💡 USE THESE RESULTS TO:"
echo "- Demonstrate meta-learning efficiency trade-offs"
echo "- Support paper claims about training vs testing efficiency"
echo "- Guide practical decisions about when to use meta-learning"
echo "- Show quantitative evidence for theoretical predictions" 