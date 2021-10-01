import torch
from torch import nn as nn
from torch import Size, Tensor

class ZEncoderNet(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
    ) -> None:
        super().__init__()

        self.z_net = torch.nn.Sequential(torch.nn.Linear(num_inputs, 100),
                            torch.nn.ReLU(),
                            torch.nn.Linear(100, 100),
                            torch.nn.ReLU(),
                            torch.nn.Linear(100, num_outputs),
                            )

    def forward(self, x: Tensor):
        return self.z_net(x)