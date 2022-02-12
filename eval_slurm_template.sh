#!/bin/bash
#SBATCH --job-name=TEMPLATE
#SBATCH --output=output_err/TEMPLATE.out
#SBATCH --error=output_err/TEMPLATE.err
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --partition PARTITION
#SBATCH --cpus-per-task=6
# ACCOUNT
###SBATCH --exclude dave
###SBATCH --exclude calculon,alexa,bmo,cortana
###SBATCH -w olivaw

source ~/.bashrc
unset PYTHONPATH

conda activate habitat-quad
cd HABITAT_REPO_PATH
export CUDA_LAUNCH_BLOCKING=1
srun python -u -m habitat_baselines.run \
    --exp-config CONFIG_YAML \
    --run-type eval