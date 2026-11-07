from collections.abc import Callable
from enum import Enum
from typing import Any

import numpy as np
import torch
from captum import attr


class _ExplainerMethods:
    def __init__(self, method, default_args) -> None:
        self.method = method
        self.default_args = default_args

    method: Callable
    default_args: dict[str, Any]


class ExplainerMethods(Enum):
    Saliency = _ExplainerMethods(attr.Saliency, {})
    InputXGradient = _ExplainerMethods(attr.InputXGradient, {})
    IntegratedGradients = _ExplainerMethods(
        attr.IntegratedGradients, {"n_steps": 10, "internal_batch_size": 5}
    )
    SmoothGrad = _ExplainerMethods(
        lambda model: attr.NoiseTunnel(attr.Saliency(model)),
        {"nt_type": "smoothgrad", "nt_samples": 15},
    )


def saliency_map(
    method: ExplainerMethods,
    model: torch.nn.Module,
    input_idx: torch.Tensor,
    target_idx: int,
    **overrides,
) -> np.ndarray:
    input_idx = input_idx.clone().detach().requires_grad_(True)
    explainer = method.value.method(model)
    attributions = explainer.attribute(
        input_idx, target=target_idx, **(method.value.default_args | overrides)
    )
    attr_value = attributions.detach().abs().sum(dim=1).squeeze(0).cpu().numpy()
    return (attr_value - attr_value.min()) / (
        attr_value.max() - attr_value.min() + 1e-12
    )
