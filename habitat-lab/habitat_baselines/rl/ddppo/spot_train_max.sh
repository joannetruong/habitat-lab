#!/bin/bash

# export GLOG_minloglevel=2
# export MAGNUM_LOG=quiet
#export PYTHONPATH=$PYTHONPATH:/nethome/mrudolph8/Documents/habspot/habitat_spot/habitat-sim/
cd ~/Documents/habspot/habitat_spot/habitat-lab/
rm /srv/share3/mrudolph8/develop/spot_rgb_imgs/imgs/*
ln -s /coc/testnvme/jtruong33/data/scene_datasets /nethome/mrudolph8/Documents/habspot/habitat_spot/habitat-lab/data/
ln -s /srv/share3/mrudolph8/data/data/URDF_demo_assets /nethome/mrudolph8/Documents/habspot/habitat_spot/habitat-lab/data/

# set -x
# python -u -m torch.distributed.launch \
#     --use_env \
#     --nproc_per_node 1 \
#     habitat_baselines/run.py \
#     --exp-config habitat_baselines/config/pointnav/ddppo_pointnav_spot_train.yaml \
#     --run-type train

python habitat_baselines/run.py --exp-config habitat_baselines/config/pointnav/ddppo_pointnav_spot_train_max.yaml --run-type train
