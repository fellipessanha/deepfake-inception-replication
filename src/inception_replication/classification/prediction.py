import torch
from torch.utils.data import DataLoader, Dataset

from inception_replication.configs import DEVICE


def prediction(
    model: torch.nn.Module,
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 0,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=DEVICE.type == "cuda",
    )

    with torch.no_grad():
        batch_results = tuple(zip(*[_process_batch(model, batch) for batch in loader]))
    if not batch_results:
        raise ValueError("dataset is empty: nothing to predict.")
    return tuple(torch.cat(result) for result in batch_results)


def _process_batch(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device = DEVICE,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    inputs, targets = batch
    probs = model(inputs.to(device)).softmax(dim=1).cpu()

    return targets, probs.argmax(dim=1), probs
