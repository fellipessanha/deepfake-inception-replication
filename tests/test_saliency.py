import numpy as np
import pytest
import torch

pytest.importorskip("captum")

from inception_replication.saliency.explainers import ExplainerMethods, saliency_map

SIZE = 16


class TwoPixel(torch.nn.Module):
    "Class 0 reads one pixel of channel 0, class 1 one pixel of channel 1: the right answer is known."

    def forward(self, x):
        return torch.stack([5 * x[:, 0, 10, 12], 5 * x[:, 1, 3, 4]], dim=1)


def _input():
    # Ones, not zeros: InputXGradient and IntegratedGradients scale by the input value.
    return torch.ones(1, 3, SIZE, SIZE)


@pytest.mark.parametrize("method", list(ExplainerMethods))
@pytest.mark.parametrize("target, pixel", [(0, (10, 12)), (1, (3, 4))])
def test_map_highlights_only_the_pixel_the_target_reads(method, target, pixel):
    result = saliency_map(method, TwoPixel(), _input(), target)

    expected = np.zeros((SIZE, SIZE))
    expected[pixel] = 1
    np.testing.assert_allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize("method", list(ExplainerMethods))
def test_map_is_finite_and_normalized(method):
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.LazyLinear(2),
    ).eval()
    model(_input())  # materialize LazyLinear

    result = saliency_map(method, model, torch.randn(1, 3, SIZE, SIZE), 1)

    assert result.shape == (SIZE, SIZE)
    assert np.isfinite(result).all()
    assert result.min() == pytest.approx(0)
    assert result.max() == pytest.approx(1)


def test_caller_input_is_left_untouched():
    x = _input()
    saliency_map(ExplainerMethods.Saliency, TwoPixel(), x, 0)
    assert not x.requires_grad
    assert x.grad is None


def test_overrides_replace_defaults():
    calls = []

    class Spy:
        def __init__(self, model):
            pass

        def attribute(self, x, target, **kwargs):
            calls.append(kwargs)
            return torch.ones_like(x)

    method = ExplainerMethods.IntegratedGradients
    original = method.value.method
    method.value.method = Spy
    try:
        saliency_map(method, TwoPixel(), _input(), 0, n_steps=2)
    finally:
        method.value.method = original

    assert calls == [{"n_steps": 2, "internal_batch_size": 5}]
