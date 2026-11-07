import torch

from inception_replication.classification.inception_model import (
    InceptionModel,
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
