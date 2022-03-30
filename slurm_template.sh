#!/bin/bash
#SBATCH --job-name=$TEMPLATE
#SBATCH --output=$LOG.out
#SBATCH --error=$LOG.err
#SBATCH --gres gpu:$GPUS
#SBATCH --nodes 1
#SBATCH --ntasks-per-node $GPUS
#SBATCH --partition $PARTITION
#SBATCH --cpus-per-task=6
# CONSTRAINT

#SBATCH --chdir $HABITAT_REPO_PATH

export CUDA_LAUNCH_BLOCKING=1
srun $CONDA_ENV -u -m habitat_baselines.run \
     --exp-config $CONFIG_YAML \
     --run-type train
