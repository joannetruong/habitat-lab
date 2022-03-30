#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from dg_util.python_utils import pytorch_util as pt_util
from habitat.utils import profiling_wrapper
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.rl.ppo.policy import Policy
from torch import Tensor
from torch import nn as nn
from torch import optim as optim

EPS_PPO = 1e-5


def get_visual_loss(outputs, label_dict, output_info):
    min_channel = 0
    visual_losses = {}
    for key, num_channels in output_info:
        outputs_on = outputs[:, min_channel : min_channel + num_channels, ...]
        if key == "reconstruction":
            labels = label_dict["rgb"].to(torch.float32) / 128.0 - 1
            visual_loss = F.l1_loss(outputs_on, labels, reduction="none")
        else:
            labels = label_dict[key]
            if key == "depth":
                visual_loss = F.l1_loss(outputs_on, labels, reduction="none")
            elif key == "semantic":
                assert labels.max() < outputs_on.shape[1]
                visual_loss = 0.25 * F.cross_entropy(
                    outputs_on, labels, reduction="none"
                )
            elif key == "surface_normals":
                # Normalize
                outputs_on = outputs_on / outputs_on.norm(dim=1, keepdim=True)
                # Cosine similarity
                visual_loss = -torch.sum(outputs_on * labels, dim=1, keepdim=True)
            else:
                raise NotImplementedError("Loss not implemented")
        visual_loss = torch.mean(visual_loss)
        if key == "surface_normals":
            visual_loss = visual_loss + 1  # just a shift so it's not negative
        visual_losses[key] = visual_loss
        min_channel += num_channels

    visual_loss_total = sum(visual_losses.values())
    visual_losses = {key: val.item() for key, val in visual_losses.items()}
    visual_loss_value = visual_loss_total.item()
    return visual_loss_total, visual_loss_value, visual_losses


def get_object_existence_loss(outputs, labels):
    loss = F.binary_cross_entropy_with_logits(outputs, labels)
    return loss


def get_visual_loss_with_rollout(batch_obs, decoder_output_info, decoder_outputs):
    labels = batch_obs.copy()
    depth_obs = torch.cat(
        [
            # Spot is cross-eyed; right is on the left on the FOV
            batch_obs["spot_right_depth"],
            batch_obs["spot_left_depth"],
        ],
        dim=2,
    )

    labels["depth"] = depth_obs.permute(0, 3, 1, 2)  # NHWC => NCHW
    if "surface_normals" in batch_obs:
        labels["surface_normals"] = batch_obs["surface_normals"].permute(
            0, 3, 1, 2
        )  # NHWC => NCHW
    return get_visual_loss(decoder_outputs, labels, decoder_output_info)


def get_egomotion_loss(actions, egomotion_pred):
    loss = F.cross_entropy(egomotion_pred, actions, reduction="none")
    return loss


def get_feature_prediction_loss(features, features_pred):
    loss = 1 - F.cosine_similarity(features, features_pred, dim=2)
    return loss


class PPO(nn.Module):
    def __init__(
        self,
        actor_critic: Policy,
        clip_param: float,
        ppo_epoch: int,
        num_mini_batch: int,
        value_loss_coef: float,
        entropy_coef: float,
        lr: Optional[float] = None,
        eps: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
        use_clipped_value_loss: bool = True,
        use_normalized_advantage: bool = True,
    ) -> None:

        super().__init__()

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        try:
            self.separate_optimizers = actor_critic.net.separate_optimizers
            self.use_visual_loss = actor_critic.net.use_visual_loss
            self.use_motion_loss = actor_critic.net.use_motion_loss

            self.update_encoder_features = actor_critic.net.update_encoder_features
            self.freeze_encoder_features = actor_critic.net.freeze_encoder_features

            self.update_visual_decoder_features = (
                actor_critic.net.update_visual_decoder_features
            )
            self.freeze_visual_decoder_features = (
                actor_critic.net.freeze_visual_decoder_features
            )

            self.update_motion_decoder_features = (
                actor_critic.net.update_motion_decoder_features
            )
            self.freeze_motion_decoder_features = (
                actor_critic.net.freeze_motion_decoder_features
            )
        except:
            self.separate_optimizers = False

        if self.separate_optimizers:
            SPLITNET_PARAMS = [
                "visual_encoder",
                "visual_fc",
                "egomotion_layer",
                "motion_model_layer",
            ]
            self.ac_parameters = [
                param
                for name, param in actor_critic.named_parameters()
                if not any([i in name for i in SPLITNET_PARAMS]) and param.requires_grad
            ]

            self.splitnet_params = [
                param
                for name, param in actor_critic.named_parameters()
                if any([i in name for i in SPLITNET_PARAMS]) and param.requires_grad
            ]
            self.splitnet_optimizer = optim.Adam(
                self.splitnet_params,
                lr=lr,
                eps=eps,
            )
        else:
            self.ac_parameters = list(
                filter(lambda p: p.requires_grad, actor_critic.parameters())
            )

        self.optimizer = optim.Adam(
            self.ac_parameters,
            lr=lr,
            eps=eps,
        )
        self.device = next(actor_critic.parameters()).device
        self.use_normalized_advantage = use_normalized_advantage

    def forward(self, *x):
        raise NotImplementedError

    def get_advantages(self, rollouts: RolloutStorage) -> Tensor:
        advantages = (
            rollouts.buffers["returns"][:-1] - rollouts.buffers["value_preds"][:-1]
        )
        if not self.use_normalized_advantage:
            return advantages

        return (advantages - advantages.mean()) / (advantages.std() + EPS_PPO)

    def update(self, rollouts: RolloutStorage) -> Tuple[float, float, float]:
        advantages = self.get_advantages(rollouts)

        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        dist_entropy_epoch = 0.0

        total_loss_epoch = 0
        visual_loss_value = 0
        egomotion_loss_value = 0
        feature_prediction_loss_value = 0
        visual_losses = {}

        for _e in range(self.ppo_epoch):
            profiling_wrapper.range_push("PPO.update epoch")
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch
            )

            for batch in data_generator:
                ## Splitnet visual and motion auxiliary losses
                if hasattr(self.actor_critic.net, "decoder_enabled") and (
                    self.use_visual_loss or self.use_motion_loss
                ):
                    if self.freeze_encoder_features:
                        visual_features = pt_util.remove_dim(
                            batch["observations"]["visual_encoder_features"][:-1],
                            1,
                        )
                    else:
                        obs = torch.cat(
                            [
                                # Spot is cross-eyed; right is on the left on the FOV
                                batch["observations"]["spot_right_depth"],
                                batch["observations"]["spot_left_depth"],
                            ],
                            dim=2,
                        )

                        obs = obs.permute(0, 3, 1, 2)  # NHWC => NCHW

                        (
                            visual_features,
                            decoder_outputs,
                            class_pred,
                        ) = self.actor_critic.net.visual_encoder(
                            obs,
                            self.use_visual_loss,
                        )

                    visual_loss_total = 0
                    egomotion_loss_total = 0
                    feature_loss_total = 0
                    if self.use_visual_loss:
                        (
                            visual_loss_total,
                            visual_loss_value,
                            visual_losses,
                        ) = get_visual_loss_with_rollout(
                            batch["observations"],
                            self.actor_critic.net.decoder_output_info,
                            decoder_outputs,
                        )

                    if self.use_motion_loss:
                        visual_features = self.actor_critic.net.visual_fc(
                            visual_features
                        )

                        visual_features = visual_features.view(
                            batch["observations"].shape[0] - 1,
                            batch["observations"].shape[1],
                            -1,
                        )
                        actions = batch["actions"][:-1].view(-1)
                        egomotion_pred = self.actor_critic.net.predict_egomotion(
                            visual_features[1:], visual_features[:-1]
                        )

                        egomotion_loss = get_egomotion_loss(actions, egomotion_pred)
                        egomotion_loss = egomotion_loss * batch["masks"][1:-1].view(-1)
                        egomotion_loss_total = 0.25 * torch.mean(egomotion_loss)
                        egomotion_loss_value = egomotion_loss_total.item()

                        action_one_hot = pt_util.get_one_hot(
                            actions, self.actor_critic.num_actions
                        )
                        next_feature_pred = self.actor_critic.net.predict_next_features(
                            visual_features[:-1], action_one_hot
                        )
                        feature_loss = get_feature_prediction_loss(
                            visual_features[1:].detach(),
                            next_feature_pred.view(visual_features[1:].shape),
                        )
                        feature_loss = feature_loss.view(-1) * batch["masks"][
                            1:-1
                        ].view(-1)
                        feature_loss_total = torch.mean(feature_loss)

                        feature_prediction_loss_value = feature_loss_total.item()
                    splitnet_total_loss = (
                        visual_loss_total + egomotion_loss_total + feature_loss_total
                    )
                    if self.separate_optimizers:
                        self.splitnet_optimizer.zero_grad()

                        self.before_backward(splitnet_total_loss)
                        splitnet_total_loss.backward()
                        self.after_backward(splitnet_total_loss)

                        self.before_step()
                        self.splitnet_optimizer.step()
                        self.after_step()
                    else:
                        self.optimizer.zero_grad()

                        self.before_backward(splitnet_total_loss)
                        splitnet_total_loss.backward()
                        self.after_backward(splitnet_total_loss)

                        self.before_step()
                        self.optimizer.step()
                        self.after_step()
                decoder_enabled = (
                    hasattr(self.actor_critic.net, "decoder_enabled")
                    and self.actor_critic.net.decoder_enabled
                )
                if decoder_enabled:
                    self.actor_critic.net.disable_decoder()

                ## PPO Policy Loss
                (values, action_log_probs, dist_entropy, _,) = self._evaluate_actions(
                    batch["observations"],
                    batch["recurrent_hidden_states"],
                    batch["prev_actions"],
                    batch["masks"],
                    batch["actions"],
                )
                ratio = torch.exp(action_log_probs - batch["action_log_probs"])
                surr1 = ratio * batch["advantages"]
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    * batch["advantages"]
                )
                action_loss = -(torch.min(surr1, surr2).mean())

                if self.use_clipped_value_loss:
                    value_pred_clipped = batch["value_preds"] + (
                        values - batch["value_preds"]
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - batch["returns"]).pow(2)
                    value_losses_clipped = (value_pred_clipped - batch["returns"]).pow(
                        2
                    )
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
                else:
                    value_loss = 0.5 * (batch["returns"] - values).pow(2)

                value_loss = value_loss.mean()
                dist_entropy = dist_entropy.mean()
                self.optimizer.zero_grad()

                total_loss = (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - dist_entropy * self.entropy_coef
                )

                self.before_backward(total_loss)
                total_loss.backward()
                self.after_backward(total_loss)

                self.before_step()
                self.optimizer.step()
                self.after_step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                total_loss_epoch += total_loss.item()

            profiling_wrapper.range_pop()  # PPO.update epoch

        num_updates = self.ppo_epoch * self.num_mini_batch

        if decoder_enabled:
            self.actor_critic.net.enable_decoder()

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        total_loss_epoch /= num_updates

        return (
            value_loss_epoch,
            action_loss_epoch,
            dist_entropy_epoch,
            total_loss_epoch,
            visual_loss_value,
            visual_losses,
            egomotion_loss_value,
            feature_prediction_loss_value,
        )

    def _evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        r"""Internal method that calls Policy.evaluate_actions.  This is used instead of calling
        that directly so that that call can be overrided with inheritence
        """
        return self.actor_critic.evaluate_actions(
            observations, rnn_hidden_states, prev_actions, masks, action
        )

    def before_backward(self, loss: Tensor) -> None:
        pass

    def after_backward(self, loss: Tensor) -> None:
        pass

    def before_step(self) -> None:
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

    def after_step(self) -> None:
        pass
