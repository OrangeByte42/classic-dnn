import torch

from typing import Any


def train_epoch(net: Any, train_iter: Any, loss: Any, trainer: Any) -> None:
    """Train the model for one epoch."""
    if isinstance(net, torch.nn.Module):
        net.train()     # Set the model to training mode
    for X, y in train_iter:
        y_hat: torch.Tensor = net(X)
        l: float = loss(y_hat, y)
        if isinstance(trainer, torch.optim.Optimizer):
            trainer.zero_grad()
            l.backward()
            trainer.step()
        else:
            l.backward()
            trainer()



