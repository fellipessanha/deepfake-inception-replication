import numpy as np
import pytest
from PIL import Image

from deepfake_detection_explainer.video_manipulation.helpers import _crop_faces

W, H = 64, 48


@pytest.fixture
def image():
    # Each pixel encodes its own coordinates (R=x, G=y), so a crop's corner pixel tells where it came from.
    pixels = np.zeros((H, W, 3), dtype=np.uint8)
    pixels[..., 0] = np.arange(W)[None, :]
    pixels[..., 1] = np.arange(H)[:, None]
    return Image.fromarray(pixels)


def test_crops_to_box(image):
    face = _crop_faces(
        image, (np.array([10.0, 5.0, 30.0, 25.0]), 0.99), cutoff_probability=0.9
    )
    assert face.size == (20, 20)
    assert face.getpixel((0, 0))[:2] == (
        10,
        5,
    )  # top-left of crop is (x1, y1) of the source


def test_rounds_float_coordinates(image):
    face = _crop_faces(
        image, (np.array([10.4, 5.6, 30.5, 25.4]), 0.99), cutoff_probability=0.9
    )
    # numpy rounds half to even: 10.4->10, 5.6->6, 30.5->30, 25.4->25
    assert face.getpixel((0, 0))[:2] == (10, 6)
    assert face.size == (20, 19)


def test_below_cutoff_returns_none(image):
    assert (
        _crop_faces(
            image, (np.array([0.0, 0.0, 10.0, 10.0]), 0.5), cutoff_probability=0.9
        )
        is None
    )


def test_uses_given_cutoff_not_default(image):
    face = _crop_faces(
        image, (np.array([0.0, 0.0, 10.0, 10.0]), 0.5), cutoff_probability=0.1
    )
    assert face is not None
