import torch

from typing import Any, Tuple
from torch import nn
from torch.nn import functional as F


### Build VGG Model Architecture ###
####################################

# Define A reshape layer which makes inuts compatible with Conv2d layers
class Reshape(nn.Module):
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X.view(-1, 1, 224, 224)

# Build GoogLeNet Inception Block
class Inception(nn.Module):
    """Inception Block for GoogLeNet."""
    def __init__(self: Any, in_channels: int, c1: int, c2: Tuple[int, int], c3: Tuple[int, int], c4: int) -> None:
        super(Inception, self).__init__()
        # Branch 1: 1x1 Convolution Layer
        self.branch1_1: nn.Conv2d = nn.Conv2d(in_channels=in_channels, out_channels=c1, kernel_size=1, padding=0, stride=1)
        # Branch 2: 1x1 Convolution Layer -> 3x3 Convolution Layer
        self.branch2_1: nn.Conv2d = nn.Conv2d(in_channels=in_channels, out_channels=c2[0], kernel_size=1, padding=0, stride=1)
        self.branch2_2: nn.Conv2d = nn.Conv2d(in_channels=c2[0], out_channels=c2[1], kernel_size=3, padding=1, stride=1)
        # Branch 3: 1x1 Convolution Layer -> 5x5 Convolution Layer
        self.branch3_1: nn.Conv2d = nn.Conv2d(in_channels=in_channels, out_channels=c3[0], kernel_size=1, padding=0, stride=1)
        self.branch3_2: nn.Conv2d = nn.Conv2d(in_channels=c3[0], out_channels=c3[1], kernel_size=5, padding=2, stride=1)
        # Branch 4: 3x3 Max Pooling Layer -> 1x1 Convolution Layer
        self.branch4_1: nn.MaxPool2d = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_2: nn.Conv2d = nn.Conv2d(in_channels=in_channels, out_channels=c4, kernel_size=1, padding=0, stride=1)

    def forward(self: Any, X: torch.Tensor) -> torch.Tensor:
        branch1_out: torch.Tensor = F.relu(self.branch1_1(X))
        branch2_out: torch.Tensor = F.relu(self.branch2_2(F.relu(self.branch2_1(X))))
        branch3_out: torch.Tensor = F.relu(self.branch3_2(F.relu(self.branch3_1(X))))
        branch4_out: torch.Tensor = F.relu(self.branch4_2(self.branch4_1(X)))
        # Concatenate the outputs of all branches along the channel dimension
        outputs: torch.Tensor = torch.cat((branch1_out, branch2_out, branch3_out, branch4_out), dim=1)
        return outputs

# Build GoogLeNet Architecture
def googlenet(in_channels: int) -> nn.Sequential:
    """Create a GoogLeNet model."""
    block1: nn.Sequential = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, padding=3, stride=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
    )
    block2: nn.Sequential = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0, stride=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
    )
    block3: nn.Sequential = nn.Sequential(
        Inception(in_channels=192, c1=64, c2=(96, 128), c3=(16, 32), c4=32),    # 64 + 128 + 32 + 32 = 256 channels
        Inception(in_channels=256, c1=128, c2=(128, 192), c3=(32, 96), c4=64),  # 128 + 192 + 96 + 64 = 480 channels
        nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
    )
    block4: nn.Sequential = nn.Sequential(
        Inception(in_channels=480, c1=192, c2=(96, 208), c3=(16, 48), c4=64),       # 192 + 208 + 48 + 64 = 512 channels
        Inception(in_channels=512, c1=160, c2=(112, 224), c3=(24, 64), c4=64),      # 160 + 224 + 64 + 64 = 512 channels
        Inception(in_channels=512, c1=128, c2=(128, 256), c3=(24, 64), c4=64),      # 128 + 256 + 64 + 64 = 512 channels
        Inception(in_channels=512, c1=112, c2=(144, 288), c3=(32, 64), c4=64),      # 112 + 288 + 64 + 64 = 528 channels
        Inception(in_channels=528, c1=256, c2=(160, 320), c3=(32, 128), c4=128),    # 256 + 320 + 128 + 128 = 832 channels
        nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
    )
    block5: nn.Sequential = nn.Sequential(
        Inception(in_channels=832, c1=256, c2=(160, 320), c3=(32, 128), c4=128),    # 256 + 320 + 128 + 128 = 832 channels
        Inception(in_channels=832, c1=384, c2=(192, 384), c3=(48, 128), c4=128),    # 384 + 384 + 128 + 128 = 1024 channels
        nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
    )
    googlenet_model: nn.Sequential = nn.Sequential(
        Reshape(),  # Reshape the input to be compatible with Conv2d layers
        block1, block2, block3, block4, block5,
        nn.Flatten(),  # Flatten the output for the fully connected layer
        nn.Linear(in_features=1024, out_features=10),  # Fully connected layer
    )
    return googlenet_model

# Initialize model parameters
def init_weights(m: nn.Module) -> None:
    """Initialize model parameters."""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)





