#!/bin/bash
#SBATCH --job-name=lr_$1
#SBATCH --output=/coc/pskynet3/jtruong33/develop/flash_results/nvidia_results/mp_419/slurm_files/lr_$1.out
#SBATCH --error=/coc/pskynet3/jtruong33/develop/flash_results/nvidia_results/mp_419/slurm_files/lr_$1.err
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 25
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu=5GB
#SBATCH --partition=long

source ~/.bashrc
conda activate nvidia
cd /coc/testnvme/jtruong33/nvidia

set -x
srun python train_future_states_actions.py \
			--obj='kettle' \
			--data='/coc/testnvme/jtruong33/data/nvidia/converted_trajs_0.035/mp/kettle_cpu/419' \
			--data-type='mp' \
			--save-dir='/coc/pskynet3/jtruong33/develop/flash_results/nvidia_results/mp_419' \
			--lr=$1 \
			--batch-size=128 \
			--state-history=100 \
			--action-horizon=100 \
			--num-layers=1 \
			--hidden-state=512 \
			--max-batches=20000 \
			--num-workers=16 \
			--grip-scale-c=5 \
			--grip-scale-o=10 \
			--actions-scale=2 \
			--relative=True