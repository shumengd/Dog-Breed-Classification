"""
Challenge CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.challenge import Challenge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

__all__ = ["Challenge"]


class Challenge(nn.Module):
    def __init__(self, dropout: float = 0.6) -> None:
        """Define model architecture."""
        super().__init__()

        # Match the source model's convolutional feature extractor.
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(64, 8, kernel_size=5, stride=2, padding=2)

        # Classification head (32 = 8*2*2)
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(32, 2)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for conv in [self.conv1, self.conv2, self.conv3]:
            nn.init.normal_(
                conv.weight,
                mean=0.0,
                std=sqrt(1.0 / (conv.kernel_size[0] ** 2 * conv.in_channels)),
            )
            nn.init.constant_(conv.bias, 0.0)
        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.fc1.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x
