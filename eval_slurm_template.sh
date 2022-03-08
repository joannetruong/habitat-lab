#!/bin/bash
#SBATCH --job-name=TEMPLATE
#SBATCH --output=TEMPLATE.out
#SBATCH --error=TEMPLATE.err
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --partition PARTITION
#SBATCH --cpus-per-task=7
#SBATCH --constraint=a40

cd HABITAT_REPO_PATH
export CUDA_LAUNCH_BLOCKING=1
srun /nethome/jtruong33/miniconda3/envs/habitat-outdoor/bin/python -u -m habitat_baselines.run \
    --exp-config CONFIG_YAML \
    --run-type eval