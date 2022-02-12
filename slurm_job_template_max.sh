#!/bin/bash
#SBATCH --job-name=TEMPLATE
#SBATCH --output=TEMPLATE.out
#SBATCH --error=TEMPLATE.err
#SBATCH --gres gpu:GPUS
#SBATCH --nodes 1
#SBATCH --ntasks-per-node GPUS
#SBATCH --partition PARTITION
#SBATCH --cpus-per-task=6
#SBATCH --constraint=rtx_6000
###SBATCH --exclude dave
###SBATCH --exclude calculon,alexa,bmo,cortana
###SBATCH -w olivaw


# export GLOG_minloglevel=2
# export MAGNUM_LOG=quiet

# export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

# set -x

source ~/.bashrc

conda activate habao
cd HABITAT_REPO_PATH
export CUDA_LAUNCH_BLOCKING=1
srun python -u -m habitat_baselines.run \
    --exp-config CONFIG_YAML \
    --run-type train
