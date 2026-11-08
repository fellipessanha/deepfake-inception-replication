import numpy as np
import pytest
import torch

pytest.importorskip("captum")

from inception_replication.saliency.explainers import ExplainerMethods, saliency_map

SIZE = 16
# GradCAM explains a named layer, which these fake models don't have: it's tested with FakeInception below.
GRADIENT_METHODS = [m for m in ExplainerMethods if m is not ExplainerMethods.GradCAM]


class TwoPixel(torch.nn.Module):
    "Class 0 reads one pixel of channel 0, class 1 one pixel of channel 1: the right answer is known."

    def forward(self, x):
        return torch.stack([5 * x[:, 0, 10, 12], 5 * x[:, 1, 3, 4]], dim=1)


def _input():
    # Ones, not zeros: InputXGradient and IntegratedGradients scale by the input value.
    return torch.ones(1, 3, SIZE, SIZE)


@pytest.mark.parametrize("method", GRADIENT_METHODS)
@pytest.mark.parametrize("target, pixel", [(0, (10, 12)), (1, (3, 4))])
def test_map_highlights_only_the_pixel_the_target_reads(method, target, pixel):
    result = saliency_map(method, TwoPixel(), _input(), target)

    expected = np.zeros((SIZE, SIZE))
    expected[pixel] = 1
    np.testing.assert_allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize("method", GRADIENT_METHODS)
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


class FakeInception(torch.nn.Module):
    """Same attribute path as InceptionModel, so GradCAM finds `inception_model.Mixed_7c`.

    Mixed_7c is a 4x4/stride-4 conv: a 16x16 input becomes 4x4 "regions". Feature 0 sums
    input channel 0 over its region, feature 1 sums channel 1; logit k is the mean of feature k.
    """

    def __init__(self):
        super().__init__()
        self.inception_model = torch.nn.Module()
        conv = torch.nn.Conv2d(3, 2, kernel_size=4, stride=4, bias=False)
        with torch.no_grad():
            conv.weight.zero_()
            conv.weight[0, 0] = 1
            conv.weight[1, 1] = 1
        self.inception_model.Mixed_7c = conv

    def forward(self, x):
        return self.inception_model.Mixed_7c(x).mean(dim=(2, 3))


# Pixel blocks of one 4x4 region each: class 0's evidence (channel 0), class 1's (channel 1).
BLOCKS = {0: (slice(4, 8), slice(8, 12)), 1: (slice(12, 16), slice(0, 4))}


def _two_block_input():
    x = torch.zeros(1, 3, SIZE, SIZE)
    for channel, (rows, cols) in BLOCKS.items():
        x[0, channel, rows, cols] = 1
    return x


@pytest.mark.parametrize("target", [0, 1])
def test_gradcam_highlights_the_region_the_target_reads(target):
    result = saliency_map(
        ExplainerMethods.GradCAM, FakeInception(), _two_block_input(), target
    )

    assert result.shape == (SIZE, SIZE)  # upsampled to the input, not left at 4x4
    assert np.isfinite(result).all()
    assert result.min() == pytest.approx(0)
    assert result.max() == pytest.approx(1)

    rows, cols = BLOCKS[target]
    peak_row, peak_col = np.unravel_index(result.argmax(), result.shape)
    assert rows.start <= peak_row < rows.stop and cols.start <= peak_col < cols.stop
    other_rows, other_cols = BLOCKS[1 - target]
    assert np.allclose(result[other_rows, other_cols], 0)  # the other class's evidence is ignored
