#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from habitat import logger
from habitat_baselines.common.aux_utils import RolloutAuxTask
from habitat_baselines.common.baseline_registry import baseline_registry
from networks.building_blocks import (Bridge, ConvBlock,
                                      ShallowUpBlockForHourglassNet)
from networks.networks import BaseResNetEncoder, ShallowVisualEncoder
from torch.distributions import Categorical, kl_divergence


def get_aux_task_class(aux_task_name: str) -> Type[nn.Module]:
    r"""Return auxiliary task class based on name.

    Args:
        aux_task_name: name of the environment.

    Returns:
        Type[nn.Module]: aux task class.
    """
    return baseline_registry.get_aux_task(aux_task_name)


@baseline_registry.register_aux_task(name="EgomotionPredictionTask")
class EgomotionPredictionTask(RolloutAuxTask):
    r"""Predict action used between two consecutive frames"""

    def __init__(self, cfg, aux_cfg, task_cfg, device, **kwargs):
        super().__init__(cfg, aux_cfg, task_cfg, device, **kwargs)
        hidden_size = aux_cfg.hidden_size
        num_actions = aux_cfg.num_actions

        self.decoder = nn.Sequential(
            # nn.Linear(2 * hidden_size, hidden_size),
            nn.Linear(4096, hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(hidden_size, num_actions),
        )
        # self.decoder = nn.Linear(2 * hidden_size + hidden_size, num_actions)
        self.criterion = nn.MSELoss(reduction="none")

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

    def get_loss(self, batch_obs):
        actions = batch_obs["actions"][:-1]
        visual_feats = batch_obs["observations"]["visual_features"]

        visual_features_curr = visual_feats[1:]
        visual_features_prev = visual_feats[:-1]
        decoder_in = torch.cat((visual_features_curr, visual_features_prev), dim=-1)
        decoder_in = decoder_in.view(-1, self.decoder[0].weight.shape[1])

        egomotion_pred = self.decoder(decoder_in)
        egomotion_loss = self.criterion(egomotion_pred, actions)

        egomotion_loss_total = 0.25 * torch.mean(egomotion_loss)
        return egomotion_loss_total


@baseline_registry.register_aux_task(name="VisualReconstructionTask")
class VisualReconstructionTask(RolloutAuxTask):
    r"""Reconstruct depth image"""

    def __init__(self, cfg, aux_cfg, task_cfg, device, **kwargs):
        super().__init__(cfg, aux_cfg, task_cfg, device, **kwargs)
        print("INIT VISUAL RECONSTRUCTION TASK")
        self.bridge = Bridge(128, 128)
        up_blocks = [
            ShallowUpBlockForHourglassNet(128, 128, upsampling_method="bilinear"),
            ShallowUpBlockForHourglassNet(128, 64, upsampling_method="bilinear"),
            ShallowUpBlockForHourglassNet(64, 32, upsampling_method="bilinear"),
            ShallowUpBlockForHourglassNet(32, 32, upsampling_method="bilinear"),
            ShallowUpBlockForHourglassNet(32, 32, upsampling_method="bilinear"),
            ShallowUpBlockForHourglassNet(32, 32, upsampling_method="bilinear"),
        ]
        self.decoder = nn.ModuleList(up_blocks)
        num_outputs = 0
        self.aux_cfg = aux_cfg
        if "depth" in aux_cfg.type:
            num_outputs += 1
        if "surface_normal" in aux_cfg.type:
            num_outputs += 3
        print("NUM OUTPUTS: ", num_outputs)
        self.out = nn.Conv2d(32, num_outputs, kernel_size=1, stride=1)

        self.criterion = nn.L1Loss(reduction="none")

    def get_depth_loss(self, batch_obs, pred):
        depth_label = torch.cat(
            [
                # Spot is cross-eyed; right is on the left on the FOV
                batch_obs["observations"]["spot_right_depth"],
                batch_obs["observations"]["spot_left_depth"],
            ],
            dim=2,
        )

        depth_label = depth_label.permute(0, 3, 1, 2)  # NHWC => NCHW
        depth_pred = pred[:, 0:1, ...]
        depth_loss = self.criterion(depth_pred, depth_label)
        depth_loss = torch.mean(depth_loss)
        return depth_loss

    def get_surface_normal_loss(self, batch_obs, pred):
        surface_normal_label = batch_obs["observations"]["surface_normals"].permute(
            0, 3, 1, 2
        )  # NHWC => NCHW
        surface_normal_pred = pred[:, 1:4, ...]
        surface_normal_pred = surface_normal_pred / surface_normal_pred.norm(
            dim=1, keepdim=True
        )
        # Cosine similarity
        surface_normal_loss = -torch.sum(
            surface_normal_pred * surface_normal_label, dim=1, keepdim=True
        )
        return surface_normal_loss

    def get_loss(self, batch_obs):
        visual_loss = 0
        visual_features = batch_obs["observations"]["visual_features"]
        x = self.bridge(visual_features)

        for i, block in enumerate(self.decoder, 1):
            x = block(x)

        pred = self.out(x)
        if "depth" in self.aux_cfg.type:
            visual_loss += self.get_depth_loss(batch_obs, pred)
        if "surface_normal" in self.aux_cfg.type:
            visual_loss += self.get_surface_normal_loss(batch_obs, pred) + 1
        visual_loss = torch.mean(visual_loss)

        return visual_loss
