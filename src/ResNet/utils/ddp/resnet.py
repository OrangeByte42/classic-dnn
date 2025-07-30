import torch

from typing import Any, Union, List
from torch import nn
from torch.nn import functional as F


### Build VGG Model Architecture ###
####################################

# Define A reshape layer which makes inuts compatible with Conv2d layers
class Reshape(nn.Module):
    def forward(self: Any, X: torch.Tensor) -> torch.Tensor:
        return X.view(-1, 1, 224, 224)

# Define A Identity layer which does nothing
class Identity(nn.Module):
    def forward(self: Any, X: torch.Tensor) -> torch.Tensor:
        return X

# Build ResNet-18 Architecture
class Residual18(nn.Module):
    """Residual Block for ResNet."""
    def __init__(self: Any, in_channels: int, out_channels: int, use_1x1conv: bool = False, strides: int = 1) -> None:
        super(Residual18, self).__init__()
        self.conv1: nn.Conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=3, padding=1, stride=strides)
        self.conv2: nn.Conv2d = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                            kernel_size=3, padding=1, stride=1)
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(num_features=out_channels)
        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(num_features=out_channels)
        self.res_conn: Any = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=1, stride=strides) if use_1x1conv else Identity()

    def forward(self: Any, X: torch.Tensor) -> torch.Tensor:
        Y: torch.Tensor = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        Y += self.res_conn(X)   # Residual connection
        return F.relu(Y)

def resnet18_block(in_channels: int, out_channels: int, num_residuals: int, first_block: bool = False) -> nn.Sequential:
    """Build a ResNet-18 block with multiple residual sub-blocks."""
    blks: List[Residual18] = []
    for i in range(num_residuals):
        if not first_block and i == 0:
            blks.append(Residual18(in_channels=in_channels, out_channels=out_channels, use_1x1conv=True, strides=2))
        else:
            blks.append(Residual18(in_channels=out_channels, out_channels=out_channels))
    return blks

def resnet18(in_channels: int) -> nn.Sequential:
    """Create ResNet-18 model."""
    block1: nn.Sequential = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, padding=3, stride=2),
        nn.BatchNorm2d(num_features=64), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
    )
    block2: nn.Sequential = resnet18_block(in_channels=64, out_channels=64, num_residuals=2, first_block=True)
    block3: nn.Sequential = resnet18_block(in_channels=64, out_channels=128, num_residuals=2)
    block4: nn.Sequential = resnet18_block(in_channels=128, out_channels=256, num_residuals=2)
    block5: nn.Sequential = resnet18_block(in_channels=256, out_channels=512, num_residuals=2)
    # Build the ResNet-18 model
    resnet18: nn.Sequential = nn.Sequential(
        Reshape(),
        *block1, *block2, *block3, *block4, *block5,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(in_features=512, out_features=10)
    )
    return resnet18

# Build ResNet-50 Architecture
class Residual50(nn.Module):
    """Residual Block for ResNet."""
    def __init__(self: Any, in_channels: int, mid_channels: int, out_channels: int, use_1x1conv: bool = False, strides: int = 1) -> None:
        super(Residual50, self).__init__()
        self.conv1: nn.Conv2d = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels,
                                            kernel_size=1, padding=0, stride=1)
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(num_features=mid_channels)
        self.conv2: nn.Conv2d = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                                            kernel_size=3, padding=1, stride=strides)
        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(num_features=mid_channels)
        self.conv3: nn.Conv2d = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels,
                                            kernel_size=1, padding=0, stride=1)
        self.bn3: nn.BatchNorm2d = nn.BatchNorm2d(num_features=out_channels)

        self.res_conn: Any = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=1, padding=0, stride=strides) if use_1x1conv else Identity()

    def forward(self: Any, X: torch.Tensor) -> torch.Tensor:
        Y: torch.Tensor = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        Y += self.res_conn(X)   # Residual connection
        return F.relu(Y)

def resnet50_block(in_channels: int, mid_channels: int, out_channels: int, num_residuals: int, first_block: bool = False) -> nn.Sequential:
    """Build a ResNet-50 block with multiple residual sub-blocks."""
    blks: List[Residual50] = []
    for i in range(num_residuals):
        if i == 0:
            blks.append(Residual50(in_channels=in_channels, mid_channels=mid_channels, out_channels=out_channels, use_1x1conv=True,
                                    strides=1 if first_block else 2))
        else:
            blks.append(Residual50(in_channels=out_channels, mid_channels=mid_channels, out_channels=out_channels))
    return blks

def resnet50(in_channels: int) -> nn.Sequential:
    """Create ResNet-50 model."""
    block1: nn.Sequential = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, padding=3, stride=2),
        nn.BatchNorm2d(num_features=64), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
    )
    block2: nn.Sequential = resnet50_block(in_channels=64, mid_channels=64, out_channels=256, num_residuals=3, first_block=True)
    block3: nn.Sequential = resnet50_block(in_channels=256, mid_channels=128, out_channels=512, num_residuals=4)
    block4: nn.Sequential = resnet50_block(in_channels=512, mid_channels=256, out_channels=1024, num_residuals=6)
    block5: nn.Sequential = resnet50_block(in_channels=1024, mid_channels=512, out_channels=2048, num_residuals=3)
    # Build the ResNet-50 model
    resnet50: nn.Sequential = nn.Sequential(
        Reshape(),
        *block1, *block2, *block3, *block4, *block5,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(in_features=2048, out_features=10)
    )
    return resnet50

# Initialize model parameters
def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)





