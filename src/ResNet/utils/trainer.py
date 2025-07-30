import torch

from typing import Any, Tuple
from d2l.torch import Animator
from utils.utils import get_right_prediction_count, evaluate_accuracy


def _train_epoch(net: Any, train_iter: Any, loss: Any, trainer: Any, device: Any) -> Tuple[float, float]:
    """Train the model for one epoch."""
    if isinstance(net, torch.nn.Module):
        net.train()     # Set the model to training mode
    metric: torch.Tensor = torch.zeros(3)   # [loss, correct predictions, total predictions]
    for X, y in train_iter:
        X, y = X.to(device), y.to(device)
        y_hat: torch.Tensor = net(X)
        l: float = loss(y_hat, y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
        with torch.no_grad():
            # torch.tensor create a new tensor in CPU defaultly
            metric += torch.tensor([float(l) * y.numel(), get_right_prediction_count(y_hat, y), y.numel()])

    return metric[0] / metric[2], metric[1] / metric[2]  # Return average loss and accuracy

def train(net: Any, train_iter: Any, test_iter: Any,
            loss: Any, num_epochs: int, trainer: Any, device: Any) -> None:
    """Train the model."""
    animator: Animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.0, 2.5],
                                    legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics: Tuple[float, float] = _train_epoch(net, train_iter, loss, trainer, device)
        test_acc: float = evaluate_accuracy(test_iter, net, device)
        animator.add(epoch + 1, train_metrics + (test_acc,))

