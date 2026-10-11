"""
Target CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.target import Target
"""

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["Target"]


class Target(nn.Module):
    def __init__(self) -> None:
        """Define model architecture."""
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 5, stride=2, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 64, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(64, 8, 5, stride=2, padding=2)
        self.fc_1 = nn.Linear(32, 2)

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize model weights."""
        torch.manual_seed(42)
        for conv in [self.conv1, self.conv2, self.conv3]:
            nn.init.normal_(
                conv.weight, mean=0, std=sqrt(1 / (5 * 5 * conv.in_channels))
            )
            nn.init.constant_(conv.bias, 0)

        nn.init.normal_(self.fc_1.weight, mean=0, std=sqrt(1 / 32))
        nn.init.constant_(self.fc_1.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        x = self.fc_1(x)
        return x
