import time
from typing import Dict

import cv2
import numpy as np
import torch
from torch import nn as nn

DEBUGGING = False


class SimpleCNN(nn.Module):
    r"""A Simple 3-Conv CNN followed by a fully connected layer

    Takes in observations and produces an embedding of the rgb and/or depth components

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
    """

    def __init__(
        self,
        observation_space,
        output_size,
    ):
        super().__init__()
        self.count = 0
        self.debug_prefix = f"{time.time() * 1e7:.0f}"[-5:]
        # HACK: Never use RGB for policies.
        # if "rgb" in observation_space.spaces:
        #     self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
        # else:
        #     self._n_input_rgb = 0
        self._n_input_rgb = 0

        # Ensure both the single camera AND two camera setup is NOT being used
        self.using_one_camera = "depth" in observation_space.spaces
        self.using_two_cameras = any(
            [k.endswith("_depth") for k in observation_space.spaces.keys()]
        )
        assert not (self.using_one_camera and self.using_two_cameras)

        # Ensure both eyes are being used if at all
        if self.using_two_cameras:
            assert all(
                [
                    i in observation_space.spaces
                    for i in ["spot_left_depth", "spot_right_depth"]
                ]
            )

        if self.using_one_camera or self.using_two_cameras:
            self._n_input_depth = 1
        else:
            self._n_input_depth = 0

        # kernel size for different CNN layers
        self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]

        # strides for different CNN layers
        self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]

        if self._n_input_rgb > 0:
            cnn_dims = np.array(
                observation_space.spaces["rgb"].shape[:2], dtype=np.float32
            )
        elif self._n_input_depth > 0:
            depth_key = "depth" if self.using_one_camera else "spot_left_depth"
            height, width = observation_space.spaces[depth_key].shape[:2]
            if self.using_two_cameras:
                width *= 2
            cnn_dims = np.array([height, width], dtype=np.float32)

        if self.is_blind:
            self.cnn = nn.Sequential()
        else:
            for kernel_size, stride in zip(
                self._cnn_layers_kernel_size, self._cnn_layers_stride
            ):
                cnn_dims = self._conv_output_dim(
                    dimension=cnn_dims,
                    padding=np.array([0, 0], dtype=np.float32),
                    dilation=np.array([1, 1], dtype=np.float32),
                    kernel_size=np.array(kernel_size, dtype=np.float32),
                    stride=np.array(stride, dtype=np.float32),
                )

            self.cnn = nn.Sequential(
                nn.Conv2d(
                    in_channels=self._n_input_rgb + self._n_input_depth,
                    out_channels=32,
                    kernel_size=self._cnn_layers_kernel_size[0],
                    stride=self._cnn_layers_stride[0],
                ),
                nn.ReLU(True),
                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=self._cnn_layers_kernel_size[1],
                    stride=self._cnn_layers_stride[1],
                ),
                nn.ReLU(True),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=32,
                    kernel_size=self._cnn_layers_kernel_size[2],
                    stride=self._cnn_layers_stride[2],
                ),
                #  nn.ReLU(True),
                nn.Flatten(),
                nn.Linear(32 * cnn_dims[0] * cnn_dims[1], output_size),
                nn.ReLU(True),
            )

        self.layer_init()

    def _conv_output_dim(self, dimension, padding, dilation, kernel_size, stride):
        r"""Calculates the output height and width based on the input
        height and width to the convolution layer.

        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(
                        (
                            (
                                dimension[i]
                                + 2 * padding[i]
                                - dilation[i] * (kernel_size[i] - 1)
                                - 1
                            )
                            / stride[i]
                        )
                        + 1
                    )
                )
            )
        return tuple(out_dimension)

    def layer_init(self):
        for layer in self.cnn:  # type: ignore
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain("relu"))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth == 0

    def forward(self, observations: Dict[str, torch.Tensor]):
        cnn_input = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations.float() / 255.0  # normalize RGB
            cnn_input.append(rgb_observations)

        if self._n_input_depth > 0:
            if self.using_one_camera:
                depth_observations = observations["depth"]
            elif self.using_two_cameras:
                depth_observations = torch.cat(
                    [
                        # Spot is cross-eyed; right is on the left on the FOV
                        observations["spot_right_depth"],
                        observations["spot_left_depth"],
                    ],
                    dim=2,
                )
            else:
                raise Exception("Not implemented")

            # Save images to disk for debugging
            if DEBUGGING:
                img = None
                h, w, c = depth_observations.shape[1:]
                for c_idx in range(c):
                    new_img = depth_observations[0][:, :, c_idx].cpu().numpy()
                    if img is None:
                        img = new_img
                    else:
                        img = np.hstack([img, new_img])
                img = (img * 255).astype(np.uint8)
                out_path = f"{self.debug_prefix}_{self.count:04}.png"
                cv2.imwrite(out_path, img)
                print("Saved visual observations to", out_path)
                self.count += 1
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            cnn_input.append(depth_observations)

        cnn_inputs = torch.cat(cnn_input, dim=1)

        return self.cnn(cnn_inputs)
