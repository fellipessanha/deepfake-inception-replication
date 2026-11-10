import numpy as np
import pytest
from PIL import Image

pytest.importorskip("captum")

from inception_replication.saliency.vlm_input import vlm_input
from test_saliency import FakeInception


def test_outputs_are_at_image_resolution_with_a_full_probability_record():
    model = FakeInception().eval()
    model.class_names = ["fake", "original"]
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, (40, 60, 3), dtype=np.uint8))  # H=40, W=60

    lit, overlay, record = vlm_input(model, image)

    assert lit.shape == overlay.shape == (40, 60, 3)  # back at the original size, not 299x299
    probs = record["probabilities"]
    assert list(probs) == model.class_names
    assert sum(probs.values()) == pytest.approx(1, abs=1e-3)
    assert record["predicted"] == max(probs, key=probs.get)
    x1, y1, x2, y2 = record["gradcam_box"]
    assert 0 <= x1 <= x2 <= 1000 and 0 <= y1 <= y2 <= 1000
