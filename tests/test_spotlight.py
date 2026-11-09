import numpy as np
import pytest
import torch

from inception_replication.saliency.spotlight import spotlight

SIZE = 40


def _square_map():
    "Salient 10x10 square at rows 10-19, cols 20-29; everything else zero."
    saliency = np.zeros((SIZE, SIZE))
    saliency[10:20, 20:30] = 1
    return saliency


def _red_image():
    raw = np.zeros((SIZE, SIZE, 3))
    raw[..., 0] = 0.9
    return raw


def test_salient_region_keeps_its_colour_and_the_rest_is_dimmed_grey():
    raw = _red_image()
    out, _ = spotlight(raw, _square_map(), keep=100 / SIZE**2, dim=0.25, feather=1)

    np.testing.assert_allclose(out[15, 25], raw[15, 25], atol=1e-2)  # centre of the square
    np.testing.assert_allclose(out[0, 0], [0.3 * 0.25] * 3, atol=1e-6)  # far corner: mean 0.3, dimmed


def test_box_is_the_square_in_qwen_coords():
    _, box = spotlight(_red_image(), _square_map(), keep=100 / SIZE**2)
    assert box == [500, 250, 725, 475]  # x 20..29, y 10..19 over 40 px, times 1000


def test_spotlight_row_adds_one_column_after_the_explanation():
    pytest.importorskip("captum")
    from inception_replication.saliency.visualize import plot_explanation_with_spotlight

    maps = {"Saliency": _square_map(), "GradCAM": _square_map()}
    axes = plot_explanation_with_spotlight(torch.tensor(_red_image()), maps, "pred: fake")

    assert len(axes) == len(maps) + 2
    assert axes[-1].get_title() == "GradCAM spotlight"
    assert len(axes[-1].patches) == 1  # the box
