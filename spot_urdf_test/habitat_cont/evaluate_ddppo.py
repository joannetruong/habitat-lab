import json
from habitat_cont.rl.resnet_policy import (
    PointNavResNetPolicy,
)

from collections import OrderedDict, defaultdict

import argparse

import random
import numpy as np
import torch
from gym import spaces
from gym.spaces.dict_space import Dict as SpaceDict
from PIL import Image

import time
import cv2
import os
import subprocess

import yaml

DEVICE = torch.device("cuda")
LOG_FILENAME = "exp.navigation.log"
MAX_DEPTH = 10


def load_model(weights_path, dim_actions):
    depth_256_space = SpaceDict({
        'depth': spaces.Box(low=0., high=1., shape=(256,256,1)),
        'pointgoal_with_gps_compass': spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(2,),
            dtype=np.float32,
        )
    })

    action_space = spaces.Box(
        np.array([float('-inf'),float('-inf')]), np.array([float('inf'),float('inf')])
    )
    action_distribution = 'gaussian'

    model = PointNavResNetPolicy(
        observation_space=depth_256_space,
        action_space=action_space,
        hidden_size=512,
        rnn_type='LSTM',
        num_recurrent_layers=2,
        backbone='resnet50',
        normalize_visual_inputs=False,
        action_distribution=action_distribution,
        dim_actions=dim_actions
    )
    model.to(torch.device(DEVICE))

    state_dict = OrderedDict()
    with open(weights_path, 'r') as f:
        state_dict = json.load(f)   
    # state_dict = torch.load(weights_path, map_location=DEVICE) 
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
