import torch

from torch.utils import data
from typing import Any, List


def load_tensor_array(tensor_array: List[torch.Tensor], batch_size: int, is_train: bool = True) -> data.DataLoader:
    """Load a tensor array into a DataLoader."""
    dataset: data.TensorDataset = data.TensorDataset(*tensor_array)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train)

def evaluate_loss(net: Any, data_iter: Any, loss: Any) -> float:
    """Evaluate model loss in specified dataset."""
    metric: torch.Tensor = torch.zeros(2)   # sum of loss, number of samples
    if isinstance(net, torch.nn.Module):
        net.eval()  # Set the model to evaluation mode
    for X, y in data_iter:
        out: torch.Tensor = net(X)
        l: torch.Tensor = loss(out, y.reshape(out.shape))
        metric += torch.tensor([l.sum(), l.numel()])
    return metric[0] / metric[1]  # Return average loss




