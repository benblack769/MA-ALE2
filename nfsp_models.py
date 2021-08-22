from functools import partial
from math import sqrt

import torch
from torch import nn as nn, Tensor
from torch.nn import init
import torch.nn.functional as F
import numpy as np

class Dueling(nn.Module):
    """ The dueling branch used in all nets that use dueling-dqn. """
    def __init__(self, value_branch, advantage_branch):
        super().__init__()
        self.flatten = nn.Flatten()
        self.value_branch = value_branch
        self.advantage_branch = advantage_branch

    def forward(self, x, advantages_only=False):
        x = self.flatten(x)
        advantages = self.advantage_branch(x)
        if advantages_only:
            return advantages

        value = self.value_branch(x)
        return value + (advantages - torch.mean(advantages, dim=1, keepdim=True))


class ImpalaCNNResidual(nn.Module):
    """
    Simple residual block used in the large IMPALA CNN.
    """
    def __init__(self, depth, norm_func):
        super().__init__()

        self.relu = nn.ReLU()
        self.conv_0 = (nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1))
        self.conv_1 = norm_func(nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        x_ = self.conv_0(self.relu(x))
        x_ = self.conv_1(self.relu(x_))
        return x+x_

def identity(p): return p

class ImpalaCNNBlock(nn.Module):
    """
    Three of these blocks are used in the large IMPALA CNN.
    """
    def __init__(self, depth_in, depth_out, norm_func):
        super().__init__()

        self.conv = norm_func(nn.Conv2d(in_channels=depth_in, out_channels=depth_out, kernel_size=3, stride=1, padding=1))
        self.max_pool = nn.MaxPool2d(3, 2, padding=1)
        self.residual_0 = ImpalaCNNResidual(depth_out, norm_func=identity)
        self.residual_1 = ImpalaCNNResidual(depth_out, norm_func=norm_func)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.residual_0(x)
        x = self.residual_1(x)
        return x


class ImpalaCNNLarge(nn.Module):
    """
    Implementation of the large variant of the IMPALA CNN introduced in Espeholt et al. (2018).
    """
    def __init__(self, in_depth, actions, linear_layer, resolution, model_size=1, spectral_norm=False):
        super().__init__()

        norm_func = torch.nn.utils.spectral_norm if spectral_norm else identity

        self.main = nn.Sequential(
            ImpalaCNNBlock(in_depth, 16*model_size, norm_func=identity),
            ImpalaCNNBlock(16*model_size, 32*model_size, norm_func=identity),
            ImpalaCNNBlock(32*model_size, 32*model_size, norm_func=norm_func),
            nn.ReLU(),
            nn.Flatten(),
        )

        shape = self.main(torch.zeros(1, in_depth, resolution[0], resolution[1])).shape
        assert shape[0] == 1
        self.linear = linear_layer(shape[1], 256)
        # print(shape[1])
        # print(model_size)
        # assert shape[1] == 32*np.prod(model_size)
        #
        # self.dueling = Dueling(
        #     nn.Sequential(linear_layer(shape[2]*shape[3]*32*model_size, 256),
        #                   nn.ReLU(),
        #                   linear_layer(256, 1)),
        #     nn.Sequential(linear_layer(shape[2]*shape[3]*32*model_size, 256),
        #                   nn.ReLU(),
        #                   linear_layer(256, actions))
        # )

    def forward(self, x, advantages_only=False):
        f = self.main(x)
        l = self.linear(f)
        # print(f.shape)
        return torch.relu(l)
        # return self.dueling(f, advantages_only=advantages_only)


if __name__ == "__main__":
    # test out net to make sure it at least can be created
    largenet = ImpalaCNNLarge(4, 18, nn.Linear, (84, 84), model_size=2)
    out = largenet(torch.zeros(2,4,84,84))
    print(out)
    print(out.shape)
