import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Subset

from inception_replication.classification.face_folder import FaceFolder
from inception_replication.classification.inception_model import InceptionModel
from inception_replication.classification.prediction import prediction
from inception_replication.configs import CLASS_NAMES, DEVICE


def get_report(
    target_values: torch.Tensor,
    prediction_values: torch.Tensor,
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


def run_pipeline(
    test_dir: str | Path,
    limit: int | None = None,
    batch_size: int = 32,
    seed: int = 0,
) -> tuple[str, pd.DataFrame]:
    dataset = FaceFolder(test_dir)
    if limit:
        # Seeded shuffle: same subset every run, and every class represented.
        generator = torch.Generator().manual_seed(seed)
        dataset = Subset(
            dataset, torch.randperm(len(dataset), generator=generator)[:limit].tolist()
        )

    model = InceptionModel().to(DEVICE).eval()
    targets, predictions, probabilities = prediction(model, dataset, batch_size)

    report = get_report(prediction_values=predictions, target_values=targets)
    frame = predictions_report_dataframe(targets, predictions, probabilities)
    return report, frame


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict the test set and print the classification report."
    )
    parser.add_argument(
        "test_dir", help="ImageFolder root with one subfolder per class"
    )
    parser.add_argument(
        "-o",
        "--output",
        help="write per-image predictions to this CSV (skipped if omitted)",
    )
    parser.add_argument(
        "--limit", type=int, help="predict only a seeded random subset of this size"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    report, frame = run_pipeline(args.test_dir, args.limit, args.batch_size)
    print(report)
    if args.output:
        frame.to_csv(args.output, index=False)
        print(f"\npredictions written to {args.output}")


if __name__ == "__main__":
    _main()
