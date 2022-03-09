#!/bin/bash
#SBATCH --job-name=$TEMPLATE
#SBATCH --output=$TEMPLATE.out
#SBATCH --error=$TEMPLATE.err
#SBATCH --gres gpu:$GPUS
#SBATCH --nodes 1
#SBATCH --ntasks-per-node $GPUS
#SBATCH --partition $PARTITION
#SBATCH --cpus-per-task=6

#SBATCH --chdir $GOOGLE_REPO_PATH

export CUDA_LAUNCH_BLOCKING=1
srun /nethome/jtruong33/miniconda3/envs/habitat-outdoor/bin/python -u -m habitat_baselines.run \
     --exp-yaml $CONFIG_YAML \
     --run-type train
