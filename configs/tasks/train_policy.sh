#!/bin/bash
#SBATCH --job-name=eval_spot_kinematic
#SBATCH --output=eval_spot_kinematic.out
#SBATCH --error=eval_spot_kinematic.err
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --partition long
#SBATCH --cpus-per-task=6
###SBATCH --constraint=rtx_6000
###SBATCH --exclude calculon,alexa,bmo,cortana
###SBATCH -w olivaw


# export GLOG_minloglevel=2
# export MAGNUM_LOG=quiet

# export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

# set -x

source ~/.bashrc
unset PYTHONPATH

conda activate habitat-spot
cd /coc/testnvme/jtruong33/habitat_spot/habitat-lab
export CUDA_LAUNCH_BLOCKING=1
srun python -u -m habitat_baselines.run \
    --exp-config habitat_baselines/config/pointnav/ddppo_pointnav_spot_kinematic_eval.yaml \
    --run-type eval
#    TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS 5
