import numpy as np
import pytest
from PIL import Image

pytest.importorskip("captum")

from inception_replication.vlm_interpretation import face_explanation
from inception_replication.vlm_interpretation.prompts import SystemPrompts
from test_saliency import FakeInception


def test_vlm_gets_face_overlay_and_record_and_its_text_is_returned(monkeypatch):
    calls = []

    def fake_generate(vlm, processor, frame, overlay, system_prompt, record):
        calls.append((frame, overlay, system_prompt, record))
        return "text"

    monkeypatch.setattr(face_explanation, "generate_llm_output", fake_generate)
    model = FakeInception().eval()
    model.class_names = ["fake", "original"]
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, (40, 60, 3), dtype=np.uint8))

    lit, overlay, record, explanation = face_explanation.explain_face(
        model, None, None, image, SystemPrompts.Claude.value
    )

    assert explanation == "text"
    [(frame, sent_overlay, prompt, sent_record)] = calls
    assert sent_record is record and record["predicted"] in model.class_names
    assert prompt == SystemPrompts.Claude.value
    assert frame.size == sent_overlay.size == (60, 40)
    assert lit.shape == overlay.shape == (40, 60, 3)
