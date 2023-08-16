#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import math
import os
import re
import shutil
import tarfile
from collections import defaultdict
from io import BytesIO
from typing import (
    Any,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch
from gym.spaces import Box
from PIL import Image
from torch import Size, Tensor
from torch import nn as nn

from habitat import logger
from habitat.core.dataset import Episode
from habitat.core.utils import try_cv2_import
from habitat.utils import profiling_wrapper
from habitat.utils.visualizations.utils import images_to_video
from habitat_baselines.common.tensor_dict import DictTree, TensorDict
from habitat_baselines.common.tensorboard_utils import TensorboardWriter

cv2 = try_cv2_import()


if hasattr(torch, "inference_mode"):
    inference_mode = torch.inference_mode
else:
    inference_mode = torch.no_grad


def cosine_decay(progress: float) -> float:
    progress = min(max(progress, 0.0), 1.0)

    return (1.0 + math.cos(progress * math.pi)) / 2.0


class CustomFixedCategorical(torch.distributions.Categorical):  # type: ignore
    def sample(
        self, sample_shape: Size = torch.Size()  # noqa: B008
    ) -> Tensor:
        return super().sample(sample_shape).unsqueeze(-1)

    def log_probs(self, actions: Tensor) -> Tensor:
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
    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)

        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: Tensor) -> CustomFixedCategorical:
        x = self.linear(x)
        return CustomFixedCategorical(logits=x)


class CustomNormal(torch.distributions.normal.Normal):
    def sample(
        self, sample_shape: Size = torch.Size()  # noqa: B008
    ) -> Tensor:
        return self.rsample(sample_shape)

    def log_probs(self, actions) -> Tensor:
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self) -> Tensor:
        return super().entropy().sum(-1, keepdim=True)


class GaussianNet(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        config: "DictConfig",
    ) -> None:
        super().__init__()

        self.action_activation = config.action_activation
        self.use_softplus = config.use_softplus
        self.use_log_std = config.use_log_std
        use_std_param = config.use_std_param
        self.clamp_std = config.clamp_std

        if self.use_log_std:
            self.min_std = config.min_log_std
            self.max_std = config.max_log_std
            std_init = config.log_std_init
        elif self.use_softplus:
            inv_softplus = lambda x: math.log(math.exp(x) - 1)
            self.min_std = inv_softplus(config.min_std)
            self.max_std = inv_softplus(config.max_std)
            std_init = inv_softplus(1.0)
        else:
            self.min_std = config.min_std
            self.max_std = config.max_std
            std_init = 1.0  # initialize std value so that std ~ 1

        if use_std_param:
            self.std = torch.nn.parameter.Parameter(
                torch.randn(num_outputs) * 0.01 + std_init
            )
            num_linear_outputs = num_outputs
        else:
            self.std = None
            num_linear_outputs = 2 * num_outputs

        self.mu_maybe_std = nn.Linear(num_inputs, num_linear_outputs)
        nn.init.orthogonal_(self.mu_maybe_std.weight, gain=0.01)
        nn.init.constant_(self.mu_maybe_std.bias, 0)

        if not use_std_param:
            nn.init.constant_(self.mu_maybe_std.bias[num_outputs:], std_init)

    def forward(self, x: Tensor) -> CustomNormal:
        mu_maybe_std = self.mu_maybe_std(x).float()
        if self.std is not None:
            mu = mu_maybe_std
            std = self.std
        else:
            mu, std = torch.chunk(mu_maybe_std, 2, -1)

        if self.action_activation == "tanh":
            mu = torch.tanh(mu)

        if self.clamp_std:
            std = torch.clamp(std, self.min_std, self.max_std)
        if self.use_log_std:
            std = torch.exp(std)
        if self.use_softplus:
            std = torch.nn.functional.softplus(std)

        return CustomNormal(mu, std, validate_args=False)


def linear_decay(epoch: int, total_num_updates: int) -> float:
    r"""Returns a multiplicative factor for linear value decay

    Args:
        epoch: current epoch number
        total_num_updates: total number of

    Returns:
        multiplicative factor that decreases param value linearly
    """
    return 1 - (epoch / float(total_num_updates))


@torch.no_grad()
@profiling_wrapper.RangeContext("batch_obs")
def batch_obs(
    observations: List[DictTree],
    device: Optional[torch.device] = None,
) -> TensorDict:
    r"""Transpose a batch of observation dicts to a dict of batched
    observations.

    Args:
        observations:  list of dicts of observations.
        device: The torch.device to put the resulting tensors on.
            Will not move the tensors if None

    Returns:
        transposed dict of torch.Tensor of observations.
    """
    batch: DefaultDict[str, List] = defaultdict(list)

    for obs in observations:
        for sensor in obs:
            batch[sensor].append(torch.as_tensor(obs[sensor]))

    batch_t: TensorDict = TensorDict()

    for sensor in batch:
        batch_t[sensor] = torch.stack(batch[sensor], dim=0)

    return batch_t.map(lambda v: v.to(device))


def get_checkpoint_id(ckpt_path: str) -> Optional[int]:
    r"""Attempts to extract the ckpt_id from the filename of a checkpoint.
    Assumes structure of ckpt.ID.path .

    Args:
        ckpt_path: the path to the ckpt file

    Returns:
        returns an int if it is able to extract the ckpt_path else None
    """
    ckpt_path = os.path.basename(ckpt_path)
    nums: List[int] = [int(s) for s in ckpt_path.split(".") if s.isdigit()]
    if len(nums) > 0:
        return nums[-1]
    return None


def poll_checkpoint_folder(
    checkpoint_folder: str, previous_ckpt_ind: int
) -> Optional[str]:
    r"""Return (previous_ckpt_ind + 1)th checkpoint in checkpoint folder
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
    models_paths.sort(key=os.path.getmtime)
    ind = previous_ckpt_ind + 1
    if ind < len(models_paths):
        return models_paths[ind]
    return None


def generate_video(
    video_option: List[str],
    video_dir: Optional[str],
    images: List[np.ndarray],
    episode_id: Union[int, str],
    checkpoint_idx: int,
    metrics: Dict[str, float],
    tb_writer: TensorboardWriter,
    fps: int = 10,
) -> None:
    r"""Generate video according to specified information.

    Args:
        video_option: string list of "tensorboard" or "disk" or both.
        video_dir: path to target video directory.
        images: list of images to be converted to video.
        episode_id: episode id for video naming.
        checkpoint_idx: checkpoint index for video naming.
        metric_name: name of the performance metric, e.g. "spl".
        metric_value: value of metric.
        tb_writer: tensorboard writer object for uploading video.
        fps: fps for generated video.
    Returns:
        None
    """
    if len(images) < 1:
        return

    metric_strs = []
    for k, v in metrics.items():
        metric_strs.append(f"{k}={v:.2f}")

    video_name = f"episode={episode_id}-ckpt={checkpoint_idx}-" + "-".join(
        metric_strs
    )
    if "disk" in video_option:
        assert video_dir is not None
        images_to_video(images, video_dir, video_name)
    if "tensorboard" in video_option:
        tb_writer.add_video_from_np_images(
            f"episode{episode_id}", checkpoint_idx, images, fps=fps
        )


def tensor_to_depth_images(tensor: Union[torch.Tensor, List]) -> np.ndarray:
    r"""Converts tensor (or list) of n image tensors to list of n images.
    Args:
        tensor: tensor containing n image tensors
    Returns:
        list of images
    """
    images = []

    for img_tensor in tensor:
        image = img_tensor.permute(1, 2, 0).cpu().numpy() * 255
        images.append(image)

    return images


def tensor_to_bgr_images(
    tensor: Union[torch.Tensor, Iterable[torch.Tensor]]
) -> List[np.ndarray]:
    r"""Converts tensor of n image tensors to list of n BGR images.
    Args:
        tensor: tensor containing n image tensors
    Returns:
        list of images
    """
    images = []

    for img_tensor in tensor:
        img = img_tensor.permute(1, 2, 0).cpu().numpy() * 255
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)

    return images


def image_resize_shortest_edge(
    img: Tensor, size: int, channels_last: bool = False
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
    img = torch.as_tensor(img)
    no_batch_dim = len(img.shape) == 3
    if len(img.shape) < 3 or len(img.shape) > 5:
        raise NotImplementedError()
    if no_batch_dim:
        img = img.unsqueeze(0)  # Adds a batch dimension
    h, w = get_image_height_width(img, channels_last=channels_last)
    if channels_last:
        if len(img.shape) == 4:
            # NHWC -> NCHW
            img = img.permute(0, 3, 1, 2)
        else:
            # NDHWC -> NDCHW
            img = img.permute(0, 1, 4, 2, 3)

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


def center_crop(
    img: Tensor, size: Union[int, Tuple[int, int]], channels_last: bool = False
) -> Tensor:
    """Performs a center crop on an image.

    Args:
        img: the array object that needs to be resized (either batched or unbatched)
        size: A sequence (h, w) or a python(int) that you want cropped
        channels_last: If the channels are the last dimension.
    Returns:
        the resized array
    """
    h, w = get_image_height_width(img, channels_last=channels_last)

    if isinstance(size, int):
        size_tuple: Tuple[int, int] = (int(size), int(size))
    else:
        size_tuple = size
    assert len(size_tuple) == 2, "size should be (h,w) you wish to resize to"
    cropy, cropx = size_tuple

    startx = w // 2 - (cropx // 2)
    starty = h // 2 - (cropy // 2)
    if channels_last:
        return img[..., starty : starty + cropy, startx : startx + cropx, :]
    else:
        return img[..., starty : starty + cropy, startx : startx + cropx]


def get_image_height_width(
    img: Union[Box, np.ndarray, torch.Tensor], channels_last: bool = False
) -> Tuple[int, int]:
    if img.shape is None or len(img.shape) < 3 or len(img.shape) > 5:
        raise NotImplementedError()
    if channels_last:
        # NHWC
        h, w = img.shape[-3:-1]
    else:
        # NCHW
        h, w = img.shape[-2:]
    return h, w


def overwrite_gym_box_shape(box: Box, shape) -> Box:
    if box.shape == shape:
        return box
    shape = list(shape) + list(box.shape[len(shape) :])
    low = box.low if np.isscalar(box.low) else np.min(box.low)
    high = box.high if np.isscalar(box.high) else np.max(box.high)
    return Box(low=low, high=high, shape=shape, dtype=box.dtype)


def get_scene_episode_dict(episodes: List[Episode]) -> Dict:
    scene_ids = []
    scene_episode_dict = {}

    for episode in episodes:
        if episode.scene_id not in scene_ids:
            scene_ids.append(episode.scene_id)
            scene_episode_dict[episode.scene_id] = [episode]
        else:
            scene_episode_dict[episode.scene_id].append(episode)

    return scene_episode_dict


def base_plus_ext(path: str) -> Union[Tuple[str, str], Tuple[None, None]]:
    """Helper method that splits off all extension.
    Returns base, allext.
    path: path with extensions
    returns: path with all extensions removed
    """
    match = re.match(r"^((?:.*/|)[^.]+)[.]([^/]*)$", path)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def valid_sample(sample: Optional[Any]) -> bool:
    """Check whether a webdataset sample is valid.
    sample: sample to be checked
    """
    return (
        sample is not None
        and isinstance(sample, dict)
        and len(list(sample.keys())) > 0
        and not sample.get("__bad__", False)
    )


def img_bytes_2_np_array(
    x: Tuple[int, torch.Tensor, bytes]
) -> Tuple[int, torch.Tensor, bytes, np.ndarray]:
    """Mapper function to convert image bytes in webdataset sample to numpy
    arrays.
    Args:
        x: webdataset sample containing ep_id, question, answer and imgs
    Returns:
        Same sample with bytes turned into np arrays.
    """
    images = []
    img_bytes: bytes
    for img_bytes in x[3:]:
        bytes_obj = BytesIO()
        bytes_obj.write(img_bytes)
        image = np.array(Image.open(bytes_obj))
        img = image.transpose(2, 0, 1)
        img = img / 255.0
        images.append(img)
    return (*x[0:3], np.array(images, dtype=np.float32))


def create_tar_archive(archive_path: str, dataset_path: str) -> None:
    """Creates tar archive of dataset and returns status code.
    Used in VQA trainer's webdataset.
    """
    logger.info("[ Creating tar archive. This will take a few minutes. ]")

    with tarfile.open(archive_path, "w:gz") as tar:
        for file in sorted(os.listdir(dataset_path)):
            tar.add(os.path.join(dataset_path, file))


def delete_folder(path: str) -> None:
    shutil.rmtree(path)


class LagrangeInequalityCoefficient(nn.Module):
    r"""Implements a learnable lagrange coefficient for a constrained
    optimization problem.


    Given the constrained optimization problem
        min f(x)
            st. x < threshold

    The lagrangian relaxation is then the dual problem
        argmax_alpha argmin_x f(x) + alpha * (x - threshold)
            st. alpha > 0

    We can optimize the dual problem via coordinate descent as
        f(x) + [[alpha]]_sg * x - alpha * ([[x]]_sg - threshold)
    To satisfy the constraint on alpha, we use projected gradient
    descent and project alpha to be > 0 after every step.

    To enforce x > threshold, we negate x and the threshold.
    This yields the coordinate descent objective
       alpha * (threshold - [[x]]_sg) - [[alpha]]_sg * x
    """

    def __init__(
        self,
        threshold: float,
        init_alpha: float = 1.0,
        alpha_min: float = 1e-4,
        alpha_max: float = 1.0,
        greater_than: bool = False,
    ):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.full((), math.log(init_alpha)))
        self.threshold = float(threshold)
        self.log_alpha_min = math.log(alpha_min)
        self.log_alpha_max = math.log(alpha_max)
        self._greater_than = greater_than

    def project_into_bounds(self):
        r"""Projects alpha back into bounds. To be called after each optim step"""
        with torch.no_grad():
            self.log_alpha.data.clamp_(self.log_alpha_min, self.log_alpha_max)

    def forward(self):
        r"""Compute alpha. This is done to allow forward hooks to work,
        the expected entry point is ref:`lagrangian_loss`"""
        return torch.exp(self.log_alpha)

    def lagrangian_loss(self, x):
        r"""Return the coordinate ascent lagrangian loss that keeps x
        less than or greater than the threshold.
        """
        alpha = self()

        if not self._greater_than:
            return alpha.detach() * x - alpha * (x.detach() - self.threshold)
        else:
            return alpha * (self.threshold - x.detach()) - alpha.detach() * x
