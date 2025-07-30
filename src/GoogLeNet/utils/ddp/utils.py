import torch

from typing import Any, Tuple
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
from torch import distributed as dist


### Load Data ###
#################

def load_fashion_mnist(
        batch_size: int, dataset_path: str, num_dataloader_workers: int, gpu_num: int,
        rank: int, resize: Any = None) -> Tuple[DataLoader, DataLoader]:
    """Load Fashion-MNIST dataset for DDP."""
    # Prepare preprocessing
    trans: Any = ([transforms.Resize(resize)] if resize else []) + [transforms.ToTensor()]
    trans = transforms.Compose(trans)
    # Load training and test datasets
    train_dataset: datasets.FashionMNIST = datasets.FashionMNIST(root=dataset_path, train=True, transform=trans)
    test_dataset: datasets.FashionMNIST = datasets.FashionMNIST(root=dataset_path, train=False, transform=trans)
    # Create distributed samplers
    train_sampler: DistributedSampler = DistributedSampler(train_dataset, num_replicas=gpu_num, rank=rank, shuffle=True)
    test_sampler: DistributedSampler = DistributedSampler(test_dataset, num_replicas=gpu_num, rank=rank, shuffle=False)
    # Create data loaders
    # Use pin_memory=True for faster data transfer to GPU
    train_dataloader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_dataloader_workers,
                                                sampler=train_sampler, pin_memory=True)
    test_dataloader: DataLoader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_dataloader_workers,
                                                sampler=test_sampler, pin_memory=True)
    # Return data loaders
    return train_dataloader, test_dataloader

### Train Assistant ###
#######################

def evaluate_accuracy(data_iter: DataLoader, net: Any, rank: int) -> float:
    """Evaluate the accuracy of the model in specified data iterator."""
    if isinstance(net, torch.nn.Module):
        net.eval()  # Set the model to evaluation mode
    metric: torch.Tensor = torch.zeros(2, device=torch.device(rank))   # [correct predictions, total predictions]
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(torch.device(rank)), y.to(torch.device(rank))
            y_hat: torch.Tensor = net(X)
            metric[0] += (y_hat.argmax(dim=1) == y).sum()
            metric[1] += y.numel()
            # 👆 avoid unnecessary IO between CPU Memory and GPU HBM than before 👇
            # metric += torch.tensor([get_right_prediction_count(net(X), y), y.numel()], device=torch.device(rank))
    # Reduce the metrics across all processes
    dist.all_reduce(metric, op=dist.ReduceOp.SUM)
    return metric[0] / metric[1]

### Manage Checkpoints ###
##########################

def save_checkpoint(epoch: int, model: nn.Module, optimizer: Any, checkpoint_path: str) -> None:
    """Save model checkpoint (only from rank 0 process)."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

### Manage GPU ###
##################

def cleanup() -> None:
    """Cleanup the distributed environment."""
    dist.destroy_process_group()  # Destroy the process group to clean up resources
    print('Distributed environment cleaned up.')
    torch.cuda.empty_cache()  # Clear the CUDA memory cache
    print('CUDA memory cache cleared.')


# 👇 Should be deleted
# def get_right_prediction_count(y_hat: torch.Tensor, y: torch.Tensor) -> int:
#     """Count the number of correct predictions."""
#     if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
#         y_hat = y_hat.argmax(dim=1)
#     cmp = (y_hat.type(y.dtype) == y)
#     return int(cmp.type(y.dtype).sum())   # result transfered to CPU by 'float()' OP



