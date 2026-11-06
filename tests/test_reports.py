import pytest

pytest.importorskip("sklearn")  # `metrics` dependency group: skip instead of erroring without it

import torch

from inception_replication.classification.reports import (
    get_report,
    predictions_report_dataframe,
)
from inception_replication.configs import CLASS_NAMES

# 3 of the 7 classes, one mistake: a true 2 predicted as 1.
TARGETS = torch.tensor([0, 1, 2])
PREDICTIONS = torch.tensor([0, 1, 1])


def test_report_accuracy():
    text = get_report(prediction_values=PREDICTIONS, target_values=TARGETS)
    assert "accuracy: 66.67%" in text


def test_report_lists_every_class_even_when_absent():
    text = get_report(prediction_values=PREDICTIONS, target_values=TARGETS)
    for name in CLASS_NAMES:
        assert name in text


def test_report_confusion_matrix_rows_are_true_classes():
    # Asymmetric on purpose: swapping targets and predictions would move the miss to the transposed cell.
    targets = torch.tensor([2])
    predictions = torch.tensor([1])
    text = get_report(prediction_values=predictions, target_values=targets)
    matrix_row = next(line for line in text.splitlines() if line.startswith("FaceSwap"))  # true class 2
    assert matrix_row.split()[1:] == ["0", "1", "0", "0", "0", "0", "0"]  # predicted class 1


def test_dataframe_columns_and_labels():
    probabilities = torch.eye(len(CLASS_NAMES))[[0, 1, 1]]
    frame = predictions_report_dataframe(
        prediction_values=PREDICTIONS,
        target_values=TARGETS,
        probabilities=probabilities,
    )
    assert list(frame.columns) == ["classe_real", "classe_predita", *CLASS_NAMES]
    assert frame["classe_real"].tolist() == ["DeepFakeDetection", "FaceShifter", "FaceSwap"]
    assert frame["classe_predita"].tolist() == ["DeepFakeDetection", "FaceShifter", "FaceShifter"]
    assert frame["FaceShifter"].tolist() == [0.0, 1.0, 1.0]
