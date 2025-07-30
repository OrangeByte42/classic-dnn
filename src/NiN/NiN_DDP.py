import os
import torch

from typing import Any
from torch import nn
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.ddp.utils import load_fashion_mnist, cleanup
from utils.ddp.nin import nin, init_weights
from utils.ddp.trainer import train


### Main Execution ###
######################

if __name__ == '__main__':
    # Setup Device and Distributed Environment
    assert torch.cuda.is_available(), 'This demo requires a GPU with CUDA support.'
    assert torch.cuda.device_count() > 1, 'This demo requires at least two GPUs.'

    print(f'torch cuda available: {torch.cuda.is_available()}')

    # world_size: int = torch.cuda.device_count()
    world_size: int = int(os.environ['WORLD_SIZE'])  # Total number of processes (GPUs) in the distributed training
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner for performance optimization

    # Initialize distributed training environment
    rank: int = int(os.environ['RANK'])  # Rank of the current process
    # Initialize the process group for distributed training
    torch.cuda.set_device(rank) # Set the current device to the rank of the process
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank, init_method='env://')
    dist.barrier()  # Synchronize all processes before starting training

    # Load Data
    mnist_path: str = os.path.join('..', '..', 'data')
    num_dataloader_workers: int = 8
    batch_size: int = 256

    train_iter, test_iter = load_fashion_mnist(batch_size=batch_size, dataset_path=mnist_path,
                                                    num_dataloader_workers=num_dataloader_workers,
                                                    gpu_num=world_size, rank=rank, resize=224)
    print(f'Rank {rank} loaded data with {len(train_iter.dataset)} training samples and {len(test_iter.dataset)} test samples.')
    dist.barrier()  # Ensure all processes have loaded the data before proceeding

    # Instantiate NiN model
    nin_model: nn.Sequential = nin(in_channels=1)
    nin_model.apply(init_weights)  # Initialize weights
    dist.barrier()  # Ensure all processes have initialized the model before proceeding

    # Setup training hyper-parameters
    num_epochs: int = 100
    lr: float = 5e-2
    net: Any = DDP(nin_model.to(torch.device(rank)), device_ids=[rank], output_device=rank)  # Move model to the appropriate device
    loss: Any = nn.CrossEntropyLoss(reduction='mean').to(torch.device(rank))  # PyTorch's CE contains softmax
    trainer: Any = torch.optim.SGD(net.parameters(), lr=lr)     # Use SGD as the optimizer

    # Train the model
    checkpoint_dir: str = os.path.join('.', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure the checkpoint directory exists
    train(net, train_iter, test_iter, loss, num_epochs, trainer, rank, checkpoint_dir=checkpoint_dir)

    # Clean up the distributed environment
    dist.barrier()  # Ensure all processes have completed training before proceeding
    cleanup()   # Clean up the distributed environment after training


