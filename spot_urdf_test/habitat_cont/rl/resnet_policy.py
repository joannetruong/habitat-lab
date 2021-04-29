#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import habitat_cont.rl.resnet as resnet
from habitat_cont.rl.running_mean_and_var import (
    RunningMeanAndVar,
)
from habitat_cont.rl.ddppo_utils import Flatten, ResizeCenterCropper
from habitat_cont.rl.rnn_state_encoder import RNNStateEncoder
from habitat_cont.rl.policy import Net, Policy


class PointNavResNetPolicy(Policy):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size=512,
        num_recurrent_layers=2,
        rnn_type="LSTM",
        resnet_baseplanes=32,
        backbone="resnet50",
        normalize_visual_inputs=False,
        obs_transform=ResizeCenterCropper(size=(256, 256)),
        dim_actions=None,
        action_distribution='categorical',
    ):
        if dim_actions is None:
            dim_actions = action_space.n
        super().__init__(
            PointNavResNetNet(
                observation_space=observation_space,
                action_space=action_space,
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                resnet_baseplanes=resnet_baseplanes,
                normalize_visual_inputs=normalize_visual_inputs,
                obs_transform=obs_transform,
                action_distribution=action_distribution,
                dim_actions=dim_actions,
            ),
            dim_actions=dim_actions,
            action_distribution=action_distribution
        )


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        observation_space,
        baseplanes=32,
        ngroups=32,
        spatial_size=128,
        make_backbone=None,
        normalize_visual_inputs=False,
        obs_transform=ResizeCenterCropper(size=(256, 256)),
    ):
        super().__init__()

        self.obs_transform = obs_transform
        if self.obs_transform is not None:
            observation_space = self.obs_transform.transform_observation_space(
                observation_space
            )

        if "rgb" in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
            spatial_size = observation_space.spaces["rgb"].shape[0] // 2
        else:
            self._n_input_rgb = 0

        if "depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
            spatial_size = observation_space.spaces["depth"].shape[0] // 2
        else:
            self._n_input_depth = 0

        if normalize_visual_inputs:
            self.running_mean_and_var = RunningMeanAndVar(
                self._n_input_depth + self._n_input_rgb
            )
        else:
            self.running_mean_and_var = nn.Sequential()

        if not self.is_blind:
            input_channels = self._n_input_depth + self._n_input_rgb
            self.backbone = make_backbone(input_channels, baseplanes, ngroups)

            final_spatial = int(
                spatial_size * self.backbone.final_spatial_compress
            )
            after_compression_flat_size = 2048
            num_compression_channels = int(
                round(after_compression_flat_size / (final_spatial ** 2))
            )
            self.compression = nn.Sequential(
                nn.Conv2d(
                    self.backbone.final_channels,
                    num_compression_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(1, num_compression_channels),
                nn.ReLU(True),
            )

            self.output_shape = (
                num_compression_channels,
                final_spatial,
                final_spatial,
            )

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth == 0

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, observations):
        if self.is_blind:
            return None

        cnn_input = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            cnn_input.append(rgb_observations)

        if self._n_input_depth > 0:
            depth_observations = observations["depth"]

            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)

            cnn_input.append(depth_observations)

        if self.obs_transform:
            cnn_input = [self.obs_transform(inp) for inp in cnn_input]

        x = torch.cat(cnn_input, dim=1)
        x = F.avg_pool2d(x, 2)

        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x


class PointNavResNetNet(Net):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size,
        num_recurrent_layers,
        rnn_type,
        backbone,
        resnet_baseplanes,
        normalize_visual_inputs,
        obs_transform=ResizeCenterCropper(size=(256, 256)),
        action_distribution='categorical',
        dim_actions=4
    ):
        super().__init__()

        self._n_prev_action = 32
        rnn_input_size = self._n_prev_action

        self.action_distribution = action_distribution
        if action_distribution == 'categorical':
            self.prev_action_embedding = nn.Embedding(dim_actions + 1, self._n_prev_action)
        elif action_distribution == 'dual_categorical':
            self.prev_action_embedding_linear  = nn.Embedding(dim_actions + 1, self._n_prev_action)
            self.prev_action_embedding_angular = nn.Embedding(dim_actions * 2, self._n_prev_action)
            rnn_input_size += self._n_prev_action
        elif action_distribution in ['gaussian','beta','multi_gaussian']:
            self.prev_action_embedding = nn.Linear(dim_actions, self._n_prev_action)
        else:
            raise RuntimeError('action_distribution {} not supported'.format(action_distribution))

        n_input_goal = (
            observation_space.spaces["pointgoal_with_gps_compass"
            ].shape[0]
            + 1
        )
        self.tgt_embeding = nn.Linear(n_input_goal, 32)
        rnn_input_size += 32

        self._hidden_size = hidden_size

        self.visual_encoder = ResNetEncoder(
            observation_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
            obs_transform=obs_transform,
        )

        if not self.visual_encoder.is_blind:
            self.visual_fc = nn.Sequential(
                Flatten(),
                nn.Linear(
                    np.prod(self.visual_encoder.output_shape), hidden_size
                ),
                nn.ReLU(True),
            )

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self._hidden_size) + rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        x = []
        if not self.is_blind:
            if "visual_features" in observations:
                visual_feats = observations["visual_features"]
            else:
                visual_feats = self.visual_encoder(observations)

            visual_feats = self.visual_fc(visual_feats)
            x.append(visual_feats)

        goal_observations = observations["pointgoal_with_gps_compass"
        ]
        goal_observations = torch.stack(
            [
                goal_observations[:, 0],
                torch.cos(-goal_observations[:, 1]),
                torch.sin(-goal_observations[:, 1]),
            ],
            -1,
        )

        x.append(self.tgt_embeding(goal_observations))

        if self.action_distribution == 'categorical':
            prev_actions = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().squeeze(dim=-1)
            )
            x.append(prev_actions)
        elif self.action_distribution == 'dual_categorical':
            prev_actions_linear = self.prev_action_embedding_linear(
                ((prev_actions[:,0].unsqueeze(-1).float() + 1) * masks).long().squeeze(dim=-1)
            )
            x.append(prev_actions_linear)
            prev_actions_angular = self.prev_action_embedding_angular(
                ((prev_actions[:,1].unsqueeze(-1).float() + 1) * masks).long().squeeze(dim=-1)
            )
            x.append(prev_actions_angular)
        else:
            x.append(self.prev_action_embedding(prev_actions.float()))

        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states