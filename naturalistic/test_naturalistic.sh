#!/bin/bash
#SBATCH --job-name=nat_test
#SBATCH --output=logs/nat_test_%j.log
#SBATCH --error=logs/nat_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=pli
#SBATCH --account=nam
# Load required modules
module purge
module load anaconda3/2023.3
eval "$(conda shell.bash hook)"
conda activate tensorflow


# Run the test script
python test_naturalistic.py \
    --model_dir /scratch/gpfs/mg7411/results/meta_baselines/conv6 \
    --data_dir /scratch/gpfs/mg7411/data/naturalistic/N_16/trainsize_6400_1200-300-100 \
    --output_dir /scratch/gpfs/mg7411/results/naturalistic_test/conv6 \
    --batch_size 4 