#!/bin/bash
#SBATCH --job-name=$TEMPLATE
#SBATCH --output=$LOG.out
#SBATCH --error=$LOG.err
#SBATCH --gres gpu:$GPUS
#SBATCH --nodes 1
#SBATCH --ntasks-per-node $GPUS
#SBATCH --partition $PARTITION
#SBATCH --cpus-per-task=7
#SBATCH --exclude randotron,robby
# CONSTRAINT

#SBATCH --chdir $HABITAT_REPO_PATH

export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
srun $CONDA_ENV -u -m habitat_baselines.rl.behavioral_cloning.behavioral_cloning \
     --exp-config $CONFIG_YAML
