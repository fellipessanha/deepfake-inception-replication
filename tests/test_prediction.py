import pytest
import torch
import torch.nn.functional as F

from inception_replication.classification.prediction import prediction

N_CLASSES = 7


class EchoModel(torch.nn.Module):
    """Predicts the class written in the input itself, so the expected output is known."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(
            torch.ones(())
        )  # a weight, so gradients would be tracked

    def forward(self, x):
        return F.one_hot(x[:, 0].long(), N_CLASSES).float() * self.scale


def _dataset(labels):
    return [(torch.tensor([float(label)]), label) for label in labels]


LABELS = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2]  # 10 samples: batches of 3, 3, 3, 1


def test_predictions_match_labels_in_order():
    y_true, y_pred, _ = prediction(EchoModel(), _dataset(LABELS), batch_size=3)
    assert y_true.tolist() == LABELS
    assert y_pred.tolist() == LABELS


def test_probabilities_are_softmax_of_logits():
    _, y_pred, probs = prediction(EchoModel(), _dataset(LABELS), batch_size=3)
    assert probs.shape == (len(LABELS), N_CLASSES)
    assert torch.allclose(probs.sum(dim=1), torch.ones(len(LABELS)))
    assert torch.equal(probs.argmax(dim=1), y_pred)


def test_wrong_predictions_are_reported_as_is():
    dataset = [(torch.tensor([3.0]), 0)]  # model will say 3, truth is 0
    y_true, y_pred, _ = prediction(EchoModel(), dataset)
    assert (y_true.item(), y_pred.item()) == (0, 3)


def test_no_gradients_tracked():
    _, _, probs = prediction(EchoModel(), _dataset(LABELS))
    assert not probs.requires_grad


def test_empty_dataset_raises():
    with pytest.raises(ValueError, match="empty"):
        prediction(EchoModel(), [])
        breakpoint()
