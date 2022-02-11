import torch
from torch import nn as nn
from torch import Size, Tensor
from habitat_baselines.utils.common import CustomNormal 

class ZEncoderNet(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
    ) -> None:
        super().__init__()

        # self.z_encoder = nn.Sequential(nn.Linear(num_inputs, 256),
        #                     nn.ReLU(),
        #                     nn.Linear(256, 128),
        #                     nn.ReLU(),
        #                     nn.Linear(128, num_outputs),
        #                     )
        self.z_encoder = nn.Sequential(nn.Linear(num_inputs, 100),
                         nn.ReLU(),
                         nn.Linear(100, 100),
                         nn.ReLU(),
                         nn.Linear(100, num_outputs),
                         )
        # self.z_encoder = nn.Sequential(nn.Linear(num_inputs, 1024),
        #                     nn.ReLU(),
        #                     nn.Linear(1024, 512),
        #                     nn.ReLU(),
        #                     nn.Linear(512, 256),
        #                     nn.ReLU(),
        #                     nn.Linear(256, 128),
        #                     nn.ReLU(),
        #                     nn.Linear(128, 64),
        #                     nn.ReLU(),
        #                     nn.Linear(64, num_outputs),
        #                     )
        # self.z_ins = nn.Parameter(torch.rand((3,1), requires_grad=True))
        # self.z_ins = nn.Parameter(torch.zeros((3,1)))
        # self.z_in = nn.Parameter(torch.rand(1))
        self.z_in = nn.Parameter(torch.rand(1))
        # self.z_in = torch.rand(1, device='cuda:0').requires_grad_(True)

    def forward(self, x):
        z_column = self.z_in.repeat(x.shape[0], 1)
        x_cat = torch.cat([z_column, x], dim=1)
        return self.z_encoder(x_cat)

class ZVarEncoderNet(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
    ) -> None:
        super().__init__()
        self.mu = nn.Sequential(nn.Linear(num_inputs, 256),
                                nn.Dropout(0.5),
                                nn.ReLU(),
                                nn.Linear(256, 128),
                                nn.Dropout(0.25),
                                nn.ReLU(),
                                nn.Linear(128, num_outputs),
                                )
        self.std = nn.Linear(num_inputs, num_outputs)
        self.z_in = nn.Parameter(torch.rand(1))

    def forward(self, x, use_params=False):
        # print('std: ', self.std, self.std.weight)
        # print('z_in: ', self.z_in, self.z_in.is_leaf, self.z_in.grad)
        print('Z_IN: ', self.z_in)
        z_column = self.z_in.repeat(x.shape[0], 1)
        if use_params:
            x_cat = torch.cat([z_column, x], dim=1)
        else:
            x_cat = z_column
        mu = torch.tanh(self.mu(x_cat))
        std = torch.clamp(self.std(x_cat), min=1e-6, max=1)

        return CustomNormal(mu, std)


class ZDecoderNet(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
    ) -> None:
        super().__init__()

        self.z_decoder = nn.Sequential(nn.Linear(num_inputs, 128),
                            nn.ReLU(),
                            nn.Linear(128, 256),
                            nn.ReLU(),
                            nn.Linear(256, num_outputs),
                            )

    def forward(self, x, curr_robot_id):
        return self.z_decoder(x)
