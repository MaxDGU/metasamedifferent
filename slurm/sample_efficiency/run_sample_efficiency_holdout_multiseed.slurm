#!/bin/bash
#SBATCH --job-name=ood_multiseed
#SBATCH --output=logs/ood_multiseed_%j.out
#SBATCH --error=logs/ood_multiseed_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
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

echo "🎯 MULTI-SEED OOD SAMPLE EFFICIENCY ANALYSIS"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""
echo "This experiment runs 5 seeds of the holdout OOD experiment"
echo "and performs comprehensive statistical analysis to determine"
echo "if meta-learning has significant advantages over vanilla SGD."
echo ""
echo "Configuration:"
echo "- Holdout task: scrambled"
echo "- Seeds: 42, 43, 44, 45, 46"
echo "- Methods: FOMAML, Second-Order MAML, Vanilla SGD"
echo "- Statistical analysis: t-tests, effect sizes, significance testing"
echo "- Output: Publication-quality plots with error bands"
echo ""
echo "Expected runtime: ~20 hours (4 hours per seed)"
echo "============================================"
echo ""

# Run the multi-seed analysis
python scripts/sample_efficiency_holdout_multiseed.py \
    --run_experiments \
    --skip_existing \
    --seeds 42 43 44 45 46 \
    --holdout_task scrambled \
    --methods fomaml second_order vanilla \
    --epochs 15 \
    --meta_batch_size 16 \
    --vanilla_batch_size 64 \
    --inner_lr 0.05 \
    --outer_lr 0.001 \
    --vanilla_lr 0.0001 \
    --adaptation_steps 3 \
    --val_frequency 3000 \
    --save_dir results/sample_efficiency_ood_multiseed \
    --data_dir /scratch/gpfs/mg7411/samedifferent/data/meta_h5/pb

echo ""
echo "============================================"
echo "🎉 MULTI-SEED OOD ANALYSIS COMPLETED!"
echo "End time: $(date)"
echo ""
echo "Results saved to: results/sample_efficiency_ood_multiseed/holdout_scrambled/multiseed_analysis/"
echo ""
echo "Key output files:"
echo "• ood_multiseed_comparison.png - Main comparison plot with error bands"
echo "• ood_statistical_summary.png - Statistical summary plots"
echo "• statistical_analysis_report.json - Detailed statistical results"
echo "• statistical_summary.txt - Human-readable summary"
echo ""
echo "The analysis will determine if meta-learning significantly"
echo "outperforms vanilla SGD in OOD scenarios with proper statistics."
echo "============================================" 