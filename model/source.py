"""
Source CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.source import Source
"""

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import set_random_seed


__all__ = ["Source"]


class Source(nn.Module):
    def __init__(self) -> None:
        """Define model architecture."""
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 5, stride=2, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 64, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(64, 8, 5, stride=2, padding=2)
        self.fc1 = nn.Linear(32, 8)

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize model weights."""
        set_random_seed()

        for conv in [self.conv1, self.conv2, self.conv3]:
            nn.init.normal_(
                conv.weight, mean=0, std=sqrt(1 / (5 * 5 * conv.in_channels))
            )
            nn.init.constant_(conv.bias, 0)

        nn.init.normal_(self.fc1.weight, mean=0, std=sqrt(1 / 32))
        nn.init.constant_(self.fc1.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward propagation for a batch of input examples. Pass the input array
        through layers of the model and return the output after the final layer.

        Args:
            x: array of shape (N, C, H, W)
                N = number of samples
                C = number of channels
                H = height
                W = width

        Returns:
            z: array of shape (1, # output classes)
        """
        N, C, H, W = x.shape

        # conv1, relu, pool
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # conv2, relu, pool
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # conv3, relu, pool
        x = F.relu(self.conv3(x))

        x = x.view(N, -1)
        x = self.fc1(x)
        return x
