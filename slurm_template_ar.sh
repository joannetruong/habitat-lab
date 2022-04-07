#!/bin/bash
#SBATCH --job-name=joanne_$TEMPLATE
#SBATCH --output=$LOG.out
#SBATCH --error=$LOG.err
#SBATCH --gres gpu:$GPUS
#SBATCH --nodes 1
#SBATCH --ntasks-per-node $GPUS
#SBATCH --partition learnlab
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=5GB
#SBATCH --time=72:00:00

#SBATCH --chdir $HABITAT_REPO_PATH

export CUDA_LAUNCH_BLOCKING=1
srun $CONDA_ENV -u -m habitat_baselines.run \
     --exp-config $CONFIG_YAML \
     --run-type train
