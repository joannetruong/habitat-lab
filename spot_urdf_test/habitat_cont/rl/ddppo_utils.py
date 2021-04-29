#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import glob
import numbers
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from gym.spaces import Box


class Flatten(nn.Module):
    def forward(self, x):
        # return x.view(x.size(0), -1)
        return x.reshape(x.size(0), -1)

class CustomFixedCategorical(torch.distributions.Categorical):
    def sample(self, sample_shape=torch.Size()):
        return super().sample(sample_shape).unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

class CategoricalNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)

        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = self.linear(x)
        return CustomFixedCategorical(logits=x)
        
class CustomFixedDualCategorical():
    def __init__(self, y1, y2):
        self.d1 = torch.distributions.Categorical(logits=y1)
        self.d2 = torch.distributions.Categorical(logits=y2)

    def sample(self, sample_shape=torch.Size()):
        s1 = self.d1.sample(sample_shape).unsqueeze(-1)
        s2 = self.d2.sample(sample_shape).unsqueeze(-1)
        actions = torch.cat((s1,s2),-1)
        return actions

    def log_probs(self, actions):
        a1 = actions[:,0].unsqueeze(-1)
        a2 = actions[:,1].unsqueeze(-1)
        l1 = (
            self.d1
            .log_prob(a1.squeeze(-1))
            .view(a1.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )
        l2 = (
            self.d2
            .log_prob(a2.squeeze(-1))
            .view(a2.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )
        return torch.div(torch.add(l1,l2), 2.)

    def entropy(self):
        e1 = self.d1.entropy()
        e2 = self.d2.entropy()
        return torch.div(torch.add(e1, e2), 2.)

    def mode(self):
        m1 = self.d1.probs.argmax(dim=-1, keepdim=True)
        m2 = self.d2.probs.argmax(dim=-1, keepdim=True)
        actions = torch.cat((m1,m2),-1)
        return actions

class DualCategoricalNet(nn.Module):
    def __init__(self, num_inputs, num_outputs1, num_outputs2):
        super().__init__()

        self.linear1 = nn.Linear(num_inputs, num_outputs1)
        self.linear2 = nn.Linear(num_inputs, num_outputs2)

        nn.init.orthogonal_(self.linear1.weight, gain=0.01)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.orthogonal_(self.linear2.weight, gain=0.01)
        nn.init.constant_(self.linear2.bias, 0)

    def forward(self, x):
        y1 = self.linear1(x)
        y2 = self.linear2(x)
        return CustomFixedDualCategorical(y1,y2)

class GaussianNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        
        self.mu  = nn.Linear(num_inputs, num_outputs)
        self.std = nn.Linear(num_inputs, num_outputs)

        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.constant_(self.mu.bias, 0)
        nn.init.orthogonal_(self.std.weight, gain=0.01)
        nn.init.constant_(self.std.bias, 0)

    def forward(self, x):
        mu = self.mu(x)
        std = self.std(x)

        std = torch.clamp(std, min=1e-6, max=1)

        # std = torch.clamp(std, min=-5, max=2)
        # std = torch.clamp(std, min=-5, max=0)
        # std = std.exp()

        return CustomNormal(mu, std)

class CustomNormal(torch.distributions.normal.Normal):
    def sample(self, sample_shape=torch.Size()):
        return super().rsample(sample_shape).unsqueeze(-1)

    def log_probs(self, actions):
        ret = (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )
        return ret

    def mode(self):
        return self.mean

# class MultiGaussianNet(nn.Module):
#     def __init__(self, num_inputs, num_outputs):
#         super().__init__()

#         self.num_outputs = num_outputs
        
#         # (total elements - diagonal)/2 + diagonal
#         num_outputs_std = int((num_outputs-1)*num_outputs/2)
        
#         self.mu  = nn.Linear(num_inputs, num_outputs)
#         self.cov = nn.Linear(num_inputs, num_outputs_std)

#         nn.init.orthogonal_(self.mu.weight, gain=0.01)
#         nn.init.constant_(self.mu.bias, 0)
#         nn.init.orthogonal_(self.cov.weight, gain=0.01)
#         nn.init.constant_(self.cov.bias, 0)

#     def forward(self, x):
#         mu = self.mu(x)
#         cov = self.cov(x)
#         cov = torch.clamp(cov, min=1e-6, max=1)

#         cov_mat = torch.zeros((cov.shape[0], self.num_outputs, self.num_outputs), device=cov.device)
#         tril_indices = torch.tril_indices(row=self.num_outputs, col=self.num_outputs, offset=0).to(cov.device)
#         triu_indices = torch.triu_indices(row=self.num_outputs, col=self.num_outputs, offset=0).to(cov.device)
#         cov_mat[:, triu_indices[0], triu_indices[1]] = cov
#         cov_mat[:, tril_indices[0], tril_indices[1]] = torch.transpose(cov_mat,1,2)[:, tril_indices[0], tril_indices[1]]
#         print('cov_mat', cov_mat)
#         return CustomMultiNormal(mu, cov_mat)
class MultiGaussianNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.num_outputs = num_outputs

        self.mu      = nn.Linear(num_inputs, num_outputs)
        self.log_cov = nn.Linear(num_inputs, num_outputs)

        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.constant_(self.mu.bias, 0)
        nn.init.orthogonal_(self.log_cov.weight, gain=0.01)
        nn.init.constant_(self.log_cov.bias, 0)

    def forward(self, x):
        mu = self.mu(x)
        log_cov = self.log_cov(x)
        log_cov = torch.clamp(log_cov, min=-5, max=2)
        cov = log_cov.exp()

        cov_mat = torch.zeros((cov.shape[0], self.num_outputs, self.num_outputs), device=cov.device)
        for i in range(cov_mat.shape[0]):
            cov_mat[i] = torch.diag(cov[i])

        return CustomMultiNormal(mu, cov_mat)

class CustomMultiNormal(torch.distributions.multivariate_normal.MultivariateNormal):
    def sample(self, sample_shape=torch.Size()):
        return super().rsample(sample_shape).unsqueeze(-1)

    def log_probs(self, actions):
        ret = (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )
        return ret

    def mode(self):
        return self.mean

class BetaNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        
        self.concentration1 = nn.Linear(num_inputs, num_outputs)
        self.concentration0 = nn.Linear(num_inputs, num_outputs)
        self.relu = nn.ReLU()

        nn.init.orthogonal_(self.concentration1.weight, gain=0.01)
        nn.init.constant_(self.concentration1.bias, 0)
        nn.init.orthogonal_(self.concentration0.weight, gain=0.01)
        nn.init.constant_(self.concentration0.bias, 0)

    def forward(self, x):
        # Use alpha and beta >= 1 according to [Chou et. al, 2017]
        concentration1 = torch.add(self.relu(self.concentration1(x)), 1.)
        concentration0 = torch.add(self.relu(self.concentration0(x)), 1.)
        return CustomBeta(concentration1, concentration0)

class CustomBeta(torch.distributions.beta.Beta):
    def sample(self, sample_shape=torch.Size()):
        return super().rsample(sample_shape).unsqueeze(-1)

    def log_probs(self, actions):
        ret = (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )
        return ret

    def mode(self):
        return self.mean

class ResizeCenterCropper(nn.Module):
    def __init__(self, size, channels_last: bool = False):
        r"""An nn module the resizes and center crops your input.
        Args:
            size: A sequence (w, h) or int of the size you wish to resize/center_crop.
                    If int, assumes square crop
            channels_list: indicates if channels is the last dimension
        """
        super().__init__()
        if isinstance(size, numbers.Number):
            size = (int(size), int(size))
        assert len(size) == 2, "forced input size must be len of 2 (w, h)"
        self._size = size
        self.channels_last = channels_last

    def transform_observation_space(
        self, observation_space, trans_keys=["rgb", "depth", "semantic"]
    ):
        size = self._size
        observation_space = copy.deepcopy(observation_space)
        if size:
            for key in observation_space.spaces:
                if (
                    key in trans_keys
                    and observation_space.spaces[key].shape != size
                ):
                    observation_space.spaces[key] = overwrite_gym_box_shape(
                        observation_space.spaces[key], size
                    )
        self.observation_space = observation_space
        return observation_space

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._size is None:
            return input

        return center_crop(
            image_resize_shortest_edge(
                input, max(self._size), channels_last=self.channels_last
            ),
            self._size,
            channels_last=self.channels_last,
        )


def linear_decay(epoch: int, total_num_updates: int) -> float:
    r"""Returns a multiplicative factor for linear value decay

    Args:
        epoch: current epoch number
        total_num_updates: total number of

    Returns:
        multiplicative factor that decreases param value linearly
    """
    return 1 - (epoch / float(total_num_updates))


def _to_tensor(v) -> torch.Tensor:
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)


def batch_obs(
    observations: List[Dict], device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    r"""Transpose a batch of observation dicts to a dict of batched
    observations.

    Args:
        observations:  list of dicts of observations.
        device: The torch.device to put the resulting tensors on.
            Will not move the tensors if None

    Returns:
        transposed dict of lists of observations.
    """
    batch = defaultdict(list)

    for obs in observations:
        for sensor in obs:
            batch[sensor].append(_to_tensor(obs[sensor]))

    for sensor in batch:
        batch[sensor] = (
            torch.stack(batch[sensor], dim=0)
            .to(device=device)
            .to(dtype=torch.float)
        )

    return batch


def poll_checkpoint_folder(
    checkpoint_folder: str, previous_ckpt_ind: int
) -> Optional[str]:
    r""" Return (previous_ckpt_ind + 1)th checkpoint in checkpoint folder
    (sorted by time of last modification).

    Args:
        checkpoint_folder: directory to look for checkpoints.
        previous_ckpt_ind: index of checkpoint last returned.

    Returns:
        return checkpoint path if (previous_ckpt_ind + 1)th checkpoint is found
        else return None.
    """
    assert os.path.isdir(checkpoint_folder), (
        f"invalid checkpoint folder " f"path {checkpoint_folder}"
    )
    models_paths = list(
        filter(os.path.isfile, glob.glob(checkpoint_folder + "/*"))
    )
    # models_paths.sort(key=os.path.getmtime)
    # ind = previous_ckpt_ind + 1
    # if ind < len(models_paths):
    #     return models_paths[ind]
    # return None
    models_paths = list(
        filter(lambda x: not os.path.isfile(x+'.done') and not x.endswith('.done'), glob.glob(checkpoint_folder + "/*"))
    )
    models_paths = sorted(models_paths, key=lambda x: int(x.split('.')[-2]))
    if len(models_paths) > 0:
        with open(models_paths[0]+'.done','w') as f:
            pass
        return models_paths[0]
    return None


def image_resize_shortest_edge(
    img, size: int, channels_last: bool = False
) -> torch.Tensor:
    """Resizes an img so that the shortest side is length of size while
        preserving aspect ratio.

    Args:
        img: the array object that needs to be resized (HWC) or (NHWC)
        size: the size that you want the shortest edge to be resize to
        channels: a boolean that channel is the last dimension
    Returns:
        The resized array as a torch tensor.
    """
    img = _to_tensor(img)
    no_batch_dim = len(img.shape) == 3
    if len(img.shape) < 3 or len(img.shape) > 5:
        raise NotImplementedError()
    if no_batch_dim:
        img = img.unsqueeze(0)  # Adds a batch dimension
    if channels_last:
        h, w = img.shape[-3:-1]
        if len(img.shape) == 4:
            # NHWC -> NCHW
            img = img.permute(0, 3, 1, 2)
        else:
            # NDHWC -> NDCHW
            img = img.permute(0, 1, 4, 2, 3)
    else:
        # ..HW
        h, w = img.shape[-2:]

    # Percentage resize
    scale = size / min(h, w)
    h = int(h * scale)
    w = int(w * scale)
    img = torch.nn.functional.interpolate(
        img.float(), size=(h, w), mode="area"
    ).to(dtype=img.dtype)
    if channels_last:
        if len(img.shape) == 4:
            # NCHW -> NHWC
            img = img.permute(0, 2, 3, 1)
        else:
            # NDCHW -> NDHWC
            img = img.permute(0, 1, 3, 4, 2)
    if no_batch_dim:
        img = img.squeeze(dim=0)  # Removes the batch dimension
    return img


def center_crop(img, size, channels_last: bool = False):
    """Performs a center crop on an image.

    Args:
        img: the array object that needs to be resized (either batched or unbatched)
        size: A sequence (w, h) or a python(int) that you want cropped
        channels_last: If the channels are the last dimension.
    Returns:
        the resized array
    """
    if channels_last:
        # NHWC
        h, w = img.shape[-3:-1]
    else:
        # NCHW
        h, w = img.shape[-2:]

    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    assert len(size) == 2, "size should be (h,w) you wish to resize to"
    cropx, cropy = size

    startx = w // 2 - (cropx // 2)
    starty = h // 2 - (cropy // 2)
    if channels_last:
        return img[..., starty : starty + cropy, startx : startx + cropx, :]
    else:
        return img[..., starty : starty + cropy, startx : startx + cropx]


def overwrite_gym_box_shape(box: Box, shape) -> Box:
    if box.shape == shape:
        return box
    shape = list(shape) + list(box.shape[len(shape) :])
    low = box.low if np.isscalar(box.low) else np.min(box.low)
    high = box.high if np.isscalar(box.high) else np.max(box.high)
    return Box(low=low, high=high, shape=shape, dtype=box.dtype)
