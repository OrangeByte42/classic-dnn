import torch

from typing import List, Tuple
from torch import nn


### Build VGG Model Architecture ###
####################################

# Define A reshape layer which makes inuts compatible with Conv2d layers
class Reshape(nn.Module):
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X.view(-1, 1, 224, 224)

# Build NiN Block
def nin_block(in_channels: int, out_channels: int,
                kernel_size: int, padding: int, stride: int) -> nn.Sequential:
    """Create a NiN block with a specified kernel which after continous 1x1 convolutions"""
    nin_block: nn.Sequential = nn.Sequential(
        # Specified convolution kernel
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                    kernel_size=kernel_size, padding=padding, stride=stride),
        nn.ReLU(),
        # Two 1x1 convolutions
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, padding=0, stride=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, padding=0, stride=1),
        nn.ReLU(),
    )
    return nin_block

# Build NiN Architecture
def nin(in_channels: int) -> nn.Sequential:
    """Create a Nin model whose architecture contains 4 NiN blocks and 1 Global Average Pooling layer"""
    nin_net: nn.Sequential = nn.Sequential(
        # Preprocess the input to make it compatible with Conv2d layers
        Reshape(),
        # NiN Block 1
        nin_block(in_channels=in_channels, out_channels=96, kernel_size=11, padding=0, stride=4),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # NiN Block 2
        nin_block(in_channels=96, out_channels=256, kernel_size=5, padding=2, stride=1),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # NiN Block 3
        nin_block(in_channels=256, out_channels=384, kernel_size=3, padding=1, stride=1),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # Dropout
        nn.Dropout(p=0.5),
        # NiN Block 4
        nin_block(in_channels=384, out_channels=10, kernel_size=3, padding=1, stride=1),
        nn.AdaptiveAvgPool2d((1, 1)),
        # Flatten the output
        nn.Flatten()    # Transform (N, C, H, W) to (batch_size, 10) where 10 is the number of classes
    )
    return nin_net

# Initialize model parameters
def init_weights(m: nn.Module) -> None:
    """Initialize weights of the model"""
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)





