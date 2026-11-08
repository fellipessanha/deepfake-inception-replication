from collections.abc import Callable
from enum import Enum
from typing import Any

import numpy as np
import torch
from captum import attr


def _gradcam_post_processing(
    attributions: torch.Tensor, input_idx: torch.Tensor
) -> torch.Tensor:
    return attr.LayerAttribution.interpolate(
        attributions, input_idx.shape[-2:], "bilinear"
    )


class _ExplainerMethods:
    def __init__(self, method, default_args, post_processing=None) -> None:
        self.method = method
        self.default_args = default_args
        self.post_processing = post_processing

    method: Callable
    default_args: dict[str, Any]
    post_processing: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None


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
    GradCAM = _ExplainerMethods(
        lambda model: attr.LayerGradCam(model, model.inception_model.Mixed_7c),
        {"relu_attributions": True},
        _gradcam_post_processing,
    )

    def saliency_map(
        self,
        model: torch.nn.Module,
        input_idx: torch.Tensor,
        target_idx: int,
        **overrides,
    ) -> np.ndarray:
        input_idx = input_idx.clone().detach().requires_grad_(True)
        explainer = self.value.method(model)
        attributions = explainer.attribute(
            input_idx, target=target_idx, **(self.value.default_args | overrides)
        )
        if self.value.post_processing:
            attributions = self.value.post_processing(attributions, input_idx)
        attr_value = attributions.detach().abs().sum(dim=1).squeeze(0).cpu().numpy()
        return (attr_value - attr_value.min()) / (
            attr_value.max() - attr_value.min() + 1e-12
        )


def saliency_map(
    method: ExplainerMethods,
    model: torch.nn.Module,
    input_idx: torch.Tensor,
    target_idx: int,
    **overrides,
) -> np.ndarray:
    return method.saliency_map(model, input_idx, target_idx, **overrides)
