#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc

import torch
from gym import spaces
from habitat.config import Config
from habitat.tasks.nav.nav import (
    IntegratedPointGoalGPSAndCompassSensor,
    ContextSensor,
)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.models.simple_cnn import SimpleCNN, SimpleCNNContext
from habitat_baselines.utils.common import CategoricalNet, GaussianNet
from torch import nn as nn


class Policy(nn.Module, metaclass=abc.ABCMeta):
    def __init__(
        self, net, dim_actions, action_distribution_type="categorical"
    ):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions
        self.action_distribution_type = action_distribution_type

        if action_distribution_type == "categorical":
            self.action_distribution = CategoricalNet(
                self.net.output_size, self.dim_actions
            )
        elif action_distribution_type == "gaussian":
            self.action_distribution = GaussianNet(
                self.net.output_size, self.dim_actions
            )
        else:
            ValueError(
                f"Action distribution {action_distribution_type} not supported."
            )

        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            if self.action_distribution_type == "categorical":
                action = distribution.mode()
            elif self.action_distribution_type == "gaussian":
                action = distribution.mean
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config, observation_space, action_space):
        pass


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


@baseline_registry.register_policy
class PointNavBaselinePolicy(Policy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        action_distribution_type: str = "gaussian",
        **kwargs,
    ):
        super().__init__(
            PointNavBaselineNet(  # type: ignore
                observation_space=observation_space,
                hidden_size=hidden_size,
                **kwargs,
            ),
            action_space.n,
            action_distribution_type=action_distribution_type,
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: spaces.Dict, action_space
    ):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=config.RL.PPO.hidden_size,
        )


@baseline_registry.register_policy
class PointNavContextPolicy(Policy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        context_hidden_size: int = 512,
        action_distribution_type: str = "gaussian",
        **kwargs,
    ):
        super().__init__(
            PointNavContextNet(  # type: ignore
                observation_space=observation_space,
                hidden_size=hidden_size,
                context_hidden_size=context_hidden_size,
                **kwargs,
            ),
            action_space.n,
            action_distribution_type=action_distribution_type,
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: spaces.Dict, action_space
    ):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=config.RL.PPO.hidden_size,
            context_hidden_size=config.RL.PPO.context_hidden_size,
        )


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class PointNavBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_size: int,
    ):
        super().__init__()

        if (
            IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            in observation_space.spaces
        ):
            self._n_input_goal = observation_space.spaces[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ].shape[0]

        self._hidden_size = hidden_size

        self.visual_encoder = SimpleCNN(observation_space, hidden_size)
        self.tgt_embeding_size = 512  # TODO: Don't hardcode
        self.tgt_encoder = nn.Sequential(
            nn.Linear(self._n_input_goal, self.tgt_embeding_size),
            nn.ReLU(),
            nn.Linear(self.tgt_embeding_size, self.tgt_embeding_size),
            nn.ReLU(),
        )

        self.state_encoder = build_rnn_state_encoder(
            (0 if self.is_blind else self._hidden_size)
            + self.tgt_embeding_size,
            self._hidden_size,
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
            x.append(self.visual_encoder(observations))
        if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observations:
            goal_observations = observations[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ]
            tgt_enc = self.tgt_encoder(goal_observations)
            x.append(tgt_enc)

        x_out = torch.cat(x, dim=1)
        x_out, rnn_hidden_states = self.state_encoder(
            x_out, rnn_hidden_states, masks
        )
        return x_out, rnn_hidden_states


class PointNavContextNet(PointNavBaselineNet):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN. + Map
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_size: int,
        context_hidden_size: int,
    ):
        super().__init__(
            observation_space=observation_space,
            hidden_size=hidden_size,
        )
        self.context_hidden_size = context_hidden_size
        self.context_encoder = SimpleCNNContext(
            observation_space, self.context_hidden_size
        )
        self.state_encoder = build_rnn_state_encoder(
            (0 if self.is_blind else self._hidden_size)
            + self.tgt_embeding_size
            + self.context_hidden_size,
            self._hidden_size,
        )
        print(
            f"##### USING CONTEXT, HIDDEN SIZE: {self.context_hidden_size} #####"
        )
        self.train()

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        x = []
        if not self.is_blind:
            ve = self.visual_encoder(observations)
            x.append(ve)
        if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observations:
            goal_observations = observations[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ]
            te = self.tgt_encoder(goal_observations)
            x.append(te)
        if ContextSensor.cls_uuid in observations:
            ce = self.context_encoder(observations)
            x.append(ce)
        x_out = torch.cat(x, dim=1)
        x_out, rnn_hidden_states = self.state_encoder(
            x_out, rnn_hidden_states, masks
        )
        return x_out, rnn_hidden_states
