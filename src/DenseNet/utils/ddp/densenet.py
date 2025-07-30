import torch

from typing import Any, Tuple, List
from torch import nn


### Build VGG Model Architecture ###
####################################

# Define A reshape layer which makes inuts compatible with Conv2d layers
class Reshape(nn.Module):
    def forward(self: Any, X: torch.Tensor) -> torch.Tensor:
        return X.view(-1, 1, 224, 224)

# Build Dense Block
class DenseLayer(nn.Module):
    def __init__(self: Any, in_channels: int, growth_rate: int, bn_size: int = 4, dropout: float = 0.0) -> None:
        super(DenseLayer, self).__init__()
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(in_channels)
        self.relu1: nn.ReLU = nn.ReLU(inplace=True)
        self.conv1: nn.Conv2d = nn.Conv2d(in_channels=in_channels, out_channels=bn_size * growth_rate,
                                            kernel_size=1, padding=0, stride=1)

        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2: nn.ReLU = nn.ReLU(inplace=True)
        self.conv2: nn.Conv2d = nn.Conv2d(in_channels=bn_size * growth_rate, out_channels=growth_rate,
                                            kernel_size=3, padding=1, stride=1)

        self.dropout: nn.Dropout = nn.Dropout(p=dropout)

    def forward(self: Any, X: torch.Tensor) -> torch.Tensor:
        Y: torch.Tensor = self.conv1(self.relu1(self.bn1(X)))
        Y = self.conv2(self.relu2(self.bn2(Y)))
        Y = self.dropout(Y)
        return torch.cat((X, Y), 1)  # Concatenate input and output

class DenseBlock(nn.Module):
    def __init__(self: Any, num_layers: int, in_channels: int, growth_rate: int, bn_size: int = 4, dropout: float = 0.0) -> None:
        super(DenseBlock, self).__init__()
        layers: List[nn.Module] = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size, dropout))
        self.block: nn.Sequential = nn.Sequential(*layers)

    def forward(self: Any, X: torch.Tensor) -> torch.Tensor:
        return self.block(X)

# Build Transition Layer
class Transition(nn.Module):
    def __init__(self: Any, in_channels: int, out_channels: int) -> None:
        super(Transition, self).__init__()
        self.bn: nn.BatchNorm2d = nn.BatchNorm2d(in_channels)
        self.relu: nn.ReLU = nn.ReLU(inplace=True)
        self.conv: nn.Conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=1, padding=0, stride=1)
        self.pool: nn.AvgPool2d = nn.AvgPool2d(kernel_size=2, padding=0, stride=2)

    def forward(self: Any, X: torch.Tensor) -> torch.Tensor:
        X = self.conv(self.relu(self.bn(X)))
        X = self.pool(X)
        return X

# Build DenseNet Architecture
class DenseNet(nn.Module):
    def __init__(self: Any, in_channels: int, num_init_features: int, num_classes: int, block_config: Tuple[int],
                    growth_rate: int = 32, bn_size: int = 4, dropout: int = 0.0) -> None:
        super(DenseNet, self).__init__()
        self.layers: nn.Sequential = nn.Sequential()
        self.layers.add_module('reshape', Reshape())
        # Initial Convolution Layer
        self.layers.add_module('initial_conv', nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=num_init_features,
                        kernel_size=7, padding=3, stride=2),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        ))
        # Dense Blocks and Transition Layers
        cur_features_num: int = num_init_features
        for i, num_layers in enumerate(block_config):
            cur_block: DenseBlock = DenseBlock(num_layers, cur_features_num, growth_rate, bn_size, dropout)
            self.layers.add_module(f'denseblock{i + 1}', cur_block)
            cur_features_num += num_layers * growth_rate

            if i != len(block_config) - 1:
                transition: Transition = Transition(cur_features_num, cur_features_num // 2)
                self.layers.add_module(f'transition{i + 1}', transition)
                cur_features_num //= 2
        # Final BatchNorm and Fully Connected Layer
        self.layers.add_module('final_bn', nn.BatchNorm2d(cur_features_num))
        self.layers.add_module('final_relu', nn.ReLU(inplace=True))
        self.layers.add_module('final_pool', nn.AdaptiveAvgPool2d((1, 1)))
        # Final Fully Connected Layer
        self.layers.add_module('final_flatten', nn.Flatten())
        self.layers.add_module('classifier', nn.Linear(cur_features_num, num_classes))

    def forward(self: Any, X: torch.Tensor) -> torch.Tensor:
        X = self.layers(X)
        return X

# Initialize model parameters
def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)





