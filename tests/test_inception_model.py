import pytest
import torch
from PIL import Image

from inception_replication.classification.inception_model import (
    InceptionModel,
    classify,
    predict,
    preprocess,
)


def test_model_loads_and_predicts():
    model = (
        InceptionModel().eval()
    )  # load_state_dict is strict, so a key/shape mismatch fails here
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 299, 299))
    assert out.shape == (1, 7)


def test_input_transform_matches_training():
    "Without it accuracy drops from ~92% to ~60%, and loading still succeeds silently"
    assert InceptionModel().inception_model.transform_input


class TinyModel(torch.nn.Module):
    "Logits are the per-channel means of the input: class k wins when channel k is brightest."

    class_names = ["red", "green", "blue"]

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))  # gives classify() a device to read

    def forward(self, x):
        return x.mean(dim=(2, 3)) * self.scale


def test_classify_returns_softmax_rows_on_cpu_without_gradients():
    probs = classify(TinyModel(), torch.randn(4, 3, 8, 8))
    assert probs.shape == (4, 3)
    assert torch.allclose(probs.sum(dim=1), torch.ones(4))
    assert probs.device.type == "cpu"
    assert not probs.requires_grad


@pytest.mark.parametrize("mode", ["RGBA", "L", "RGB"])
def test_preprocess_gives_three_channel_inception_input(mode):
    assert preprocess(Image.new(mode, (100, 60))).shape == (3, 299, 299)


def test_predict_labels_the_top_class():
    x = torch.zeros(3, 8, 8)
    x[1] = 1  # green brightest
    pred = predict(TinyModel(), x)
    assert pred.label == "green"
    assert pred.x.shape == (1, 3, 8, 8)
    assert list(pred.by_class()) == TinyModel.class_names
    assert sum(pred.by_class().values()) == pytest.approx(1)
