#!/bin/bash
#SBATCH --job-name=spot_urdf_kinematic
#SBATCH --output=spot_urdf_kinematic.out
#SBATCH --error=spot_urdf_kinematic.err
#SBATCH --gres gpu:8
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 8
#SBATCH --partition long
#SBATCH --cpus-per-task=6
#SBATCH --constraint=rtx_6000
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
    --exp-config habitat_baselines/config/pointnav/ddppo_pointnav_spot_kinematic_train.yaml \
    --run-type train

