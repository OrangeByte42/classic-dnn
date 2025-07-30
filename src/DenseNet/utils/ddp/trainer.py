import os
import torch

from typing import Any, Tuple
from torch import distributed as dist
from torch.amp import GradScaler, autocast

from utils.ddp.utils import evaluate_accuracy, save_checkpoint


### Model Training Procedure ###
################################

def _train_epoch(net: Any, train_iter: Any, loss: Any, trainer: Any, rank: int, scaler: GradScaler) -> Tuple[float, float]:
    """Train the model for one epoch."""
    if isinstance(net, torch.nn.Module):
        net.train()     # Set the model to training mode
    metric: torch.Tensor = torch.zeros(3, device=torch.device(rank))   # [loss, correct predictions, total predictions]
    for X, y in train_iter:
        X, y = X.to(torch.device(rank)), y.to(torch.device(rank))
        with autocast(device_type=torch.device(rank).type, enabled=True):  # Enable mixed precision
            y_hat: torch.Tensor = net(X)
            l: torch.Tensor = loss(y_hat, y)
        trainer.zero_grad()
        # l.backward()
        # trainer.step()
        scaler.scale(l).backward()  # Scale the loss for mixed precision
        scaler.step(trainer)    # Update the model parameters
        scaler.update()     # Update the scaler for the next iteration
        with torch.no_grad():
            # torch.tensor create a new tensor in CPU defaultly
            # metric += torch.tensor([float(l) * y.numel(), (y_hat.argmax(dim=1) == y).sum().item(), y.numel()], device=torch.device(rank))
            # 👇 avoid unnecessary IO between CPU Memory and GPU HBM than before 👆
            metric[0] += l * y.numel()
            metric[1] += (y_hat.argmax(dim=1) == y).sum()
            metric[2] += y.numel()
    # Reduce the metrics across all processes
    dist.all_reduce(metric, op=dist.ReduceOp.SUM)
    return metric[0] / metric[2], metric[1] / metric[2]  # Return average loss and accuracy

def train(net: Any, train_iter: Any, test_iter: Any,
            loss: Any, num_epochs: int, trainer: Any, rank: Any,
            checkpoint_dir: str) -> None:
    """Train the model."""
    scaler: GradScaler = GradScaler()  # For mixed precision training
    for epoch in range(num_epochs):
        # dist.barrier()  # Ensure all processes are synchronized before starting the epoch
        train_iter.sampler.set_epoch(epoch)
        train_metrics: Tuple[float, float] = _train_epoch(net, train_iter, loss, trainer, rank, scaler)
        # dist.barrier()  # Ensure all processes have completed the epoch before proceeding
        test_acc: float = evaluate_accuracy(test_iter, net, rank)
        if rank == 0:
            checkpoint_path: str = os.path.join(checkpoint_dir, f'densenet121_epoch_{epoch + 1}_checkpoints.pth')
            save_checkpoint(epoch + 1, net, trainer, checkpoint_path)
            print(f'Epoch {epoch + 1:0{len(str(num_epochs))}d}/{num_epochs}, '
                    f'Train Loss: {train_metrics[0]:.4f}, '
                    f'Train Accuracy: {train_metrics[1]:.4f}, '
                    f'Test Accuracy: {test_acc:.4f}')




