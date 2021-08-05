#!/bin/bash

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

ln -s /coc/testnvme/jtruong33/data/scene_datasets /nethome/mrudolph8/Documents/habspot/habitat_spot/habitat-lab/data/

set -x
python -u -m torch.distributed.launch \
    --use_env \
    --nproc_per_node 1 \
    habitat_baselines/run.py \
    --exp-config habitat_baselines/config/pointnav/ddppo_pointnav_spot_train.yaml \
    --run-type train
