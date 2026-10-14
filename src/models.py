"""Model definitions for CSCI 566 image classification assignment."""

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """Fixed CNN architecture for FashionMNIST classification.

    Architecture:
    1. Conv2d(1 -> 16, kernel_size=3, padding=1), ReLU, MaxPool2d(2, 2)
    2. Conv2d(16 -> 32, kernel_size=3, padding=1), ReLU, MaxPool2d(2, 2)
    3. Flatten (32 * 7 * 7 = 1568), Linear(1568 -> 10)
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass for SmallCNN.

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            Logits tensor of shape (batch_size, 10).
        """
        # Exercise 1: Implement the forward pass for SmallCNN
        # Pass input x through conv1, relu1, pool1, conv2, relu2, pool2, flatten, and fc.
        # Return raw logits without softmax.
        #raise NotImplementedError("Student TODO: Implement SmallCNN.forward")
        
        #convolution-ReLU-max-pooling
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        return self.fc(self.flatten(x))
