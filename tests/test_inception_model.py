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
