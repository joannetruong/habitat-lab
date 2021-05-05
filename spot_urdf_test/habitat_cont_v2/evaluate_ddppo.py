import json
from habitat.config import Config
from habitat_baselines.config.default import get_config
from habitat_baselines.rl.ddppo.policy.resnet_policy import (
    PointNavResNetPolicy,
)
# from habitat_cont_v2.rl.resnet_policy import (
#     PointNavResNetPolicy,
# )
from habitat.core.spaces import ActionSpace, EmptySpace

from collections import OrderedDict, defaultdict

import argparse

import random
import numpy as np
import torch
from gym import spaces
from gym.spaces import Dict as SpaceDict
from PIL import Image

import time
import cv2
import os
import subprocess

import yaml

DEVICE = torch.device("cpu")
LOG_FILENAME = "exp.navigation.log"
MAX_DEPTH = 10


def load_model(weights_path, dim_actions):
    depth_256_space = SpaceDict({
        'depth': spaces.Box(low=0., high=1., shape=(240,320,1)),
        'pointgoal_with_gps_compass': spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(2,),
            dtype=np.float32,
        )
    })

    if dim_actions==3:
        action_space = ActionSpace(
            {
                "linear_velocity": EmptySpace(),
                "strafe_velocity": EmptySpace(),
                "angular_velocity": EmptySpace(),
            }
        )
    else:
        action_space = ActionSpace(
            {
                "linear_velocity": EmptySpace(),
                "angular_velocity": EmptySpace(),
            }
        )
    default_config = get_config(config_paths='config/ddppo_pointnav.yaml')
    model = PointNavResNetPolicy(
        observation_space=depth_256_space,
        action_space=action_space,
        hidden_size=512,
        rnn_type='LSTM',
        num_recurrent_layers=2,
        backbone='resnet50',
        normalize_visual_inputs=False,
        force_blind_policy= False,
        policy_config=default_config.RL.POLICY,
    )
    model.to(torch.device(DEVICE))

    # state_dict = OrderedDict()
    # with open(weights_path, 'r') as f:
    #     state_dict = json.load(f)   
    state_dict = torch.load(weights_path, map_location='cpu')['state_dict'] 
    # model.load_state_dict(state_dict["state_dict"])
    model.load_state_dict(
        {
            k[len("actor_critic.") :]: torch.tensor(v)
            for k, v in state_dict.items()
            if k.startswith("actor_critic.")
        }
    )

    return model

def to_tensor(v):
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)
