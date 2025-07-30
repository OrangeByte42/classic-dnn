import torch

from typing import List, Tuple
from torch import nn


### Build VGG Model Architecture ###
####################################

class Reshape(nn.Module):
    """Make inputs compatible with Conv2d layers."""
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X.view(-1, 1, 224, 224)

def vgg_block(num_convs: int, in_channels: int, out_channels: int,
                kernel_size: int = 3, padding: int = 1, stride: int = 1,
                pool_kernel_size: int = 2, pool_stride: int = 2) -> nn.Sequential:
    """Create a VGG block with multiple convolutional layers followed by a max pooling layer."""
    # Initialize a list to hold the layers
    layers: List[nn.Module] = []
    # Add the specified number of convolutional layers
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, padding=padding, stride=stride))
        layers.append(nn.ReLU())
        in_channels = out_channels
    # Add a max pooling layer
    layers.append(nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride))
    # Return the sequential model containing the layers
    return nn.Sequential(*layers)

def vgg(in_channels: int, conv_arch: Tuple[Tuple[int, int]]) -> nn.Sequential:
    """Create a VGG network with the specified architecture.
    Args:
        in_channels (int): Number of input channels.
        conv_arch (Tuple[Tuple[int, int]]): Architecture of the convolutional layers, where each tuple contains
            the number of convolutional layers and the number of output channels for that block.
    """
    # Initialize a list to hold the blocks
    vgg_blocks: List[nn.Sequential] = []
    # Iterate over the architecture to create each block
    for (num_convs, out_channels) in conv_arch:
        vgg_blocks.append(vgg_block(num_convs, in_channels, out_channels))  # Create a VGG block and append it to the list
        in_channels = out_channels      # Update the number of input channels for the next block
    # Create a sequential model containing all the VGG blocks
    vgg_net: nn.Sequential = nn.Sequential(
        # Preprocess the input to make it compatible with Conv2d layers
        Reshape(),
        # VGG Blocks
        *vgg_blocks,
        # Prepare for Fully Connected Layers
        nn.Flatten(),
        # Fully Connected Dense Block
        nn.Linear(out_channels * 7 * 7, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        # Output Layer
        nn.Linear(4096, 10)  # Assuming 10 classes for Fashion-MNIST
    )
    # Return the complete VGG network
    return vgg_net

def init_weights(m: nn.Module) -> None:
    """Initialize weights of the model."""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)





