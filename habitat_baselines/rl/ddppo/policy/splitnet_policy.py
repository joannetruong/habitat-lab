#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from collections import OrderedDict, deque
from typing import Dict, Tuple

import numpy as np
import torch
from dg_util.python_utils import pytorch_util as pt_util
from gym import spaces
from habitat.config import Config
from habitat.tasks.nav.nav import IntegratedPointGoalGPSAndCompassSensor
from habitat_baselines.common.auxiliary_tasks import *
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
from habitat_baselines.rl.models.rnn_state_encoder import \
    build_rnn_state_encoder
from habitat_baselines.rl.ppo import Net, Policy
from networks.building_blocks import ConvBlock
from networks.networks import BaseResNetEncoder, ShallowVisualEncoder
from torch import nn as nn


@baseline_registry.register_policy
class PointNavSplitNetPolicy(Policy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        num_recurrent_layers: int = 1,
        rnn_type: str = "GRU",
        action_distribution_type: str = "gaussian",
        visual_encoder: str = "ShallowVisualEncoder",
        create_decoder: bool = True,
        decoder_output: list = ["depth"],
        separate_optimizers: bool = True,
        use_visual_loss: bool = True,
        use_motion_loss: bool = False,
        update_encoder_features: bool = True,
        freeze_encoder_features: bool = False,
        update_visual_decoder_features: bool = True,
        freeze_visual_decoder_features: bool = False,
        update_motion_decoder_features: bool = False,
        freeze_motion_decoder_features: bool = True,
        **kwargs,
    ):
        discrete_actions = action_distribution_type == "categorical"
        super().__init__(
            PointNavSplitNetNet(
                observation_space=observation_space,
                action_space=action_space,  # for previous action
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                discrete_actions=discrete_actions,
                visual_encoder=visual_encoder,
                create_decoder=create_decoder,
                decoder_output=decoder_output,
                separate_optimizers=separate_optimizers,
                use_visual_loss=use_visual_loss,
                use_motion_loss=use_motion_loss,
                update_encoder_features=update_encoder_features,
                freeze_encoder_features=freeze_encoder_features,
                update_visual_decoder_features=update_visual_decoder_features,
                freeze_visual_decoder_features=freeze_visual_decoder_features,
                update_motion_decoder_features=update_motion_decoder_features,
                freeze_motion_decoder_features=freeze_motion_decoder_features,
            ),
            dim_actions=action_space.n,  # for action distribution
            action_distribution_type=action_distribution_type,
        )
        self.action_distribution_type = action_distribution_type

    @classmethod
    def from_config(cls, config: Config, observation_space: spaces.Dict, action_space):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=config.RL.PPO.hidden_size,
            rnn_type=config.RL.DDPPO.rnn_type,
            num_recurrent_layers=config.RL.DDPPO.num_recurrent_layers,
            backbone=config.RL.DDPPO.backbone,
            normalize_visual_inputs="rgb" in observation_space.spaces,
            force_blind_policy=config.FORCE_BLIND_POLICY,
            action_distribution_type=config.RL.POLICY.action_distribution_type,
            visual_encoder=config.RL.SPLITNET.visual_encoder,
            create_decoder=config.RL.SPLITNET.create_decoder,
            decoder_output=config.RL.SPLITNET.decoder_output,
            separate_optimizers=config.RL.SPLITNET.separate_optimizers,
            use_visual_loss=config.RL.SPLITNET.use_visual_loss,
            use_motion_loss=config.RL.SPLITNET.use_motion_loss,
            e_encoder_features=config.RL.SPLITNET.update_encoder_features,
            freeze_encoder_features=config.RL.SPLITNET.freeze_encoder_features,
            update_visual_decoder_features=config.RL.SPLITNET.update_visual_decoder_features,
            freeze_visual_decoder_features=config.RL.SPLITNET.freeze_visual_decoder_features,
            update_motion_decoder_features=config.RL.SPLITNET.update_motion_decoder_features,
            freeze_motion_decoder_features=config.RL.SPLITNET.freeze_motion_decoder_features,
        )

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        if self.net.visual_encoder_type == "BaseResNetEncoder":
            visual_feats = self.net.visual_encoder(observations)
        else:
            obs = self.net.merge_spot_obs(observations)
            obs = obs.permute(0, 3, 1, 2)  # NHWC => NCHW
            visual_feats, _, _ = self.net.visual_encoder(obs, self.net.create_decoder)

        observations["visual_features"] = visual_feats
        return super().act(
            observations, rnn_hidden_states, prev_actions, masks, deterministic
        )

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        if self.net.visual_encoder_type == "BaseResNetEncoder":
            visual_feats = self.net.visual_encoder(observations)
        else:
            obs = self.net.merge_spot_obs(observations)
            obs = obs.permute(0, 3, 1, 2)  # NHWC => NCHW
            visual_feats, _, _ = self.net.visual_encoder(obs, self.net.create_decoder)
        observations["visual_features"] = visual_feats
        return super().evaluate_actions(
            observations, rnn_hidden_states, prev_actions, masks, action
        )

    def evaluate_aux_losses(self, batch, aux_tasks):
        return [task.get_loss(batch) for task in aux_tasks]


class PointNavSplitNetNet(Net):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int,
        num_recurrent_layers: int,
        rnn_type: str,
        discrete_actions: bool = True,
        visual_encoder=ShallowVisualEncoder,
        create_decoder=True,
        decoder_output=["depth"],
        separate_optimizers=True,
        use_visual_loss=True,
        use_motion_loss=False,
        update_encoder_features=True,
        freeze_encoder_features=False,
        update_visual_decoder_features=True,
        freeze_visual_decoder_features=False,
        update_motion_decoder_features=False,
        freeze_motion_decoder_features=True,
    ):
        super().__init__()

        self.separate_optimizers = separate_optimizers
        self.discrete_actions = discrete_actions
        if discrete_actions:
            self.prev_action_embedding = nn.Embedding(action_space.n + 1, 32)
        else:
            self.prev_action_embedding = nn.Linear(action_space.n, 32)

        self._n_prev_action = 32
        rnn_input_size = self._n_prev_action

        if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observation_space.spaces:
            n_input_goal = (
                observation_space.spaces[
                    IntegratedPointGoalGPSAndCompassSensor.cls_uuid
                ].shape[0]
                + 1
            )
            self.tgt_embeding = nn.Linear(n_input_goal, 32)
            rnn_input_size += 32

        self._hidden_size = hidden_size

        decoder_output_info = [
            ("reconstruction", 1),
            ("depth", 1),
            ("surface_normals", 3),
        ]

        self.decoder_output_info = [
            x for x in decoder_output_info if x[0] in decoder_output
        ]

        self.using_gray_camera = any(
            [k.endswith("_gray") for k in observation_space.spaces.keys()]
        )
        self.using_depth_camera = any(
            [k.endswith("_depth") for k in observation_space.spaces.keys()]
        )
        self.visual_encoder_type = visual_encoder
        if visual_encoder == "BaseResNetEncoder":
            self.visual_encoder = ResNetEncoder(
                observation_space,
                baseplanes=64,
                ngroups=64 // 2,
                make_backbone=getattr(resnet, "resnet18"),
                normalize_visual_inputs=False,
            )
            self.visual_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(np.prod(self.visual_encoder.output_shape), hidden_size),
                nn.ReLU(True),
            )

        else:
            self.create_decoder = create_decoder
            self.visual_encoder = ShallowVisualEncoder(
                self.decoder_output_info,
                create_decoder,
                self.using_gray_camera or self.using_depth_camera,
            )

            self.num_output_channels = self.visual_encoder.num_output_channels

            self.visual_fc = nn.Sequential(
                ConvBlock(self.num_output_channels, hidden_size),
                ConvBlock(hidden_size, hidden_size),
                nn.AvgPool2d(2, 2),
                pt_util.RemoveDim((2, 3)),
                nn.Linear(hidden_size * 4 * 4, hidden_size),
            )

            self.egomotion_layer = nn.Sequential(
                nn.Linear(2 * hidden_size, hidden_size),
                nn.ELU(inplace=True),
                nn.Linear(hidden_size, action_space.n),
            )

            self.motion_model_layer = nn.Sequential(
                nn.Linear(hidden_size + action_space.n, hidden_size),
                nn.ELU(inplace=True),
                nn.Linear(hidden_size, hidden_size),
            )

        self.state_encoder = build_rnn_state_encoder(
            (0 if self.is_blind else self._hidden_size) + rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.use_visual_loss = use_visual_loss
        self.use_motion_loss = use_motion_loss
        self.update_encoder_features = update_encoder_features
        self.freeze_encoder_features = freeze_encoder_features
        self.update_visual_decoder_features = update_visual_decoder_features
        self.freeze_visual_decoder_features = freeze_visual_decoder_features
        self.update_motion_decoder_features = update_motion_decoder_features
        self.freeze_motion_decoder_features = freeze_motion_decoder_features

        self.decoder_enabled = create_decoder
        self.decoder_outputs = None
        self.class_pred = None
        self.visual_encoder_features = None
        self.visual_features = None

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def predict_egomotion(self, visual_features_curr, visual_features_prev):
        feature_t_concat = torch.cat(
            (visual_features_curr, visual_features_prev), dim=-1
        )
        if len(feature_t_concat.shape) > 2:
            feature_t_concat = feature_t_concat.view(
                -1, self.egomotion_layer[0].weight.shape[1]
            )
        egomotion_pred = self.egomotion_layer(feature_t_concat)
        return egomotion_pred

    def predict_next_features(self, visual_features_curr, action):
        feature_shape = visual_features_curr.shape
        if len(visual_features_curr.shape) > 2:
            visual_features_curr = visual_features_curr.view(
                -1,
                self.motion_model_layer[0].weight.shape[1] - self.action_size,
            )
        if len(action.shape) > 2:
            action = action.view(-1, self.action_size)
        next_features_delta = self.motion_model_layer(
            torch.cat((visual_features_curr, action), dim=1)
        )
        next_features = visual_features_curr + next_features_delta
        next_features = next_features.view(feature_shape)
        return next_features

    def enable_decoder(self):
        self.decoder_enabled = True
        self.decoder_outputs = None

    def disable_decoder(self):
        self.decoder_enabled = False
        self.decoder_outputs = None
        pass

    def merge_spot_obs(self, observations):
        if self.using_gray_camera:
            left_obs = observations["spot_left_gray"]
            right_obs = observations["spot_right_gray"]
        elif self.using_depth_camera:
            left_obs = observations["spot_left_depth"]
            right_obs = observations["spot_right_depth"]
        obs = torch.cat(
            [
                # Spot is cross-eyed; right is on the left on the FOV
                right_obs,
                left_obs,
            ],
            dim=2,
        )
        return obs

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = []
        if not self.is_blind:
            if "visual_features" in observations:
                visual_feats = observations["visual_features"]
            else:
                if self.visual_encoder_type == "BaseResNetEncoder":
                    visual_feats = self.visual_encoder(observations)
                else:
                    obs = self.merge_spot_obs(observations)
                    obs = obs.permute(0, 3, 1, 2)  # NHWC => NCHW
                    visual_feats, _, _ = self.visual_encoder(obs, self.create_decoder)
            visual_feats = self.visual_fc(visual_feats)
            x.append(visual_feats)

        if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observations:
            goal_observations = observations[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ]
            if goal_observations.shape[1] == 2:
                # Polar Dimensionality 2
                # 2D polar transform
                goal_observations = torch.stack(
                    [
                        goal_observations[:, 0],
                        torch.cos(-goal_observations[:, 1]),
                        torch.sin(-goal_observations[:, 1]),
                    ],
                    -1,
                )
            else:
                assert goal_observations.shape[1] == 3, "Unsupported dimensionality"
                vertical_angle_sin = torch.sin(goal_observations[:, 2])
                # Polar Dimensionality 3
                # 3D Polar transformation
                goal_observations = torch.stack(
                    [
                        goal_observations[:, 0],
                        torch.cos(-goal_observations[:, 1]) * vertical_angle_sin,
                        torch.sin(-goal_observations[:, 1]) * vertical_angle_sin,
                        torch.cos(goal_observations[:, 2]),
                    ],
                    -1,
                )
            x.append(self.tgt_embeding(goal_observations))

        prev_actions = self.prev_action_embedding(prev_actions.float())

        x.append(prev_actions)

        out = torch.cat(x, dim=1)
        out, rnn_hidden_states = self.state_encoder(out, rnn_hidden_states, masks)
        return out, rnn_hidden_states
