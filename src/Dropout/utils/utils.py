import os
import torch

from torch.utils import data
from torchvision import datasets, transforms
from typing import Any, Tuple, List


def load_fashion_mnist(
        batch_size: int, dataset_path: str,
        num_dataloader_workers: int,
        resize: Any = None) -> Tuple[data.DataLoader, data.DataLoader]:
    """Load Fashion-MNIST dataset."""
    trans: Any = ([transforms.Resize(resize)] if resize else []) + [transforms.ToTensor()]
    trans = transforms.Compose(trans)

    train_data: Any = datasets.FashionMNIST(root=dataset_path, train=True, transform=trans)
    test_data: Any = datasets.FashionMNIST(root=dataset_path, train=False, transform=trans)

    train_loader: data.DataLoader = data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_dataloader_workers)
    test_loader: data.DataLoader = data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=num_dataloader_workers)

    return train_loader, test_loader

def get_right_prediction_count(y_hat: torch.Tensor, y: torch.Tensor) -> int:
    """Count the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    cmp = (y_hat.type(y.dtype) == y)
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(data_iter: data.DataLoader, net: Any) -> float:
    """Evaluate the accuracy of the model in specified data iterator."""
    if isinstance(net, torch.nn.Module):
        net.eval()  # Set the model to evaluation mode
    metric: torch.Tensor = torch.zeros(2)   # [correct predictions, total predictions]
    with torch.no_grad():
        for X, y in data_iter:
            metric += torch.tensor([get_right_prediction_count(net(X), y), y.numel()])
    return metric[0] / metric[1]

def verify_trained_model(net: Any, test_data_iter: data.DataLoader, n: int = 6) -> None:
    """Verify the trained model by showing the first n predictions."""
    from d2l.torch import get_fashion_mnist_labels
    from d2l.torch import show_images

    X, y = next(iter(test_data_iter))
    trues: str = get_fashion_mnist_labels(y)
    preds: str = get_fashion_mnist_labels(net(X).argmax(dim=1))
    title: List[str] = [f'{true}\n{pred}' for true, pred in zip(trues, preds)]
    show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=title[0:n])



