#!/bin/bash
#SBATCH --job-name=$TEMPLATE
#SBATCH --output=$LOG.out
#SBATCH --error=$LOG.err
#SBATCH --gres gpu:$GPUS
#SBATCH --nodes 1
#SBATCH --ntasks-per-node $GPUS
#SBATCH --cpus-per-task 7
#SBATCH --partition $PARTITION

#SBATCH --chdir $HABITAT_REPO_PATH

export CUDA_LAUNCH_BLOCKING=1
srun $CONDA_ENV examples/learning/habitat_baselines_test.py \
     --exp-yaml $CONFIG_YAML \
     --run-type train