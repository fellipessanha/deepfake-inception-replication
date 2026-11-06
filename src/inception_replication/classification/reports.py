import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from inception_replication.configs import CLASS_NAMES


def get_report(
    prediction_values: torch.Tensor,
    target_values: torch.Tensor,
    class_names=CLASS_NAMES,
):
    accuracy = (prediction_values == target_values).float().mean()
    labels = [idx for idx, _ in enumerate(class_names)]
    table = classification_report(
        target_values,
        prediction_values,
        labels=labels,
        target_names=class_names,
        zero_division=0,
        digits=4,
    )
    matrix = pd.DataFrame(
        confusion_matrix(target_values, prediction_values, labels=labels),
        index=class_names,
        columns=class_names,
    )

    return "\n\n".join([f"accuracy: {accuracy:.2%}", table, matrix.to_string()])


def predictions_report_dataframe(
    target_values: torch.Tensor,
    prediction_values: torch.Tensor,
    probabilities: torch.Tensor,
    class_names=CLASS_NAMES,
):
    frame = pd.DataFrame(probabilities.numpy(), columns=class_names)
    frame.insert(0, "classe_predita", [class_names[i] for i in prediction_values])
    frame.insert(0, "classe_real", [class_names[i] for i in target_values])

    return frame
