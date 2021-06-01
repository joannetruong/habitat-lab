#!/bin/bash
#SBATCH --job-name=JOB_NAME
#SBATCH --output=JOB_NAME_eval.out
#SBATCH --error=JOB_NAME_eval.err
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --partition overcap
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=3
#SBATCH --account overcap
###SBATCH --exclude calculon,claptrap,alexa,bmo,cortana,oppy,walle,ava

source ~/.bashrc
conda activate habitat-contn
cd /srv/share3/jtruong33/develop/habitat-cont/habitat-lab
srun python -u -m habitat_baselines.run \
    --exp-config YAML_PATH \
    --run-type eval