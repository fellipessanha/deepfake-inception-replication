import os
from dataclasses import dataclass

import torch
from PIL.Image import Image
from torchvision import models

from inception_replication.classification.face_folder import TRANSFORM_COMPOSITION
from inception_replication.configs import DEVICE, PRETRAINED_INCEPTION_PATH, CLASS_NAMES


class InceptionModel(torch.nn.Module):
    def __init__(
        self,
        available_classes: list[str] = CLASS_NAMES,
        inception_specs: os.PathLike = PRETRAINED_INCEPTION_PATH,
    ) -> None:
        super().__init__()
        # No ImageNet weights: the checkpoint below overwrites every parameter anyway.
        self.class_names = available_classes
        self.inception_model: models.Inception3 = models.inception_v3(
            weights=None, aux_logits=False, init_weights=False, transform_input=True
        )
        self.inception_model.fc = torch.nn.Linear(
            self.inception_model.fc.in_features, len(self.class_names)
        )
        _ = self.load_state_dict(
            torch.load(inception_specs, map_location=DEVICE, weights_only=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inception_model(x)


def load_model(weights: os.PathLike | str = PRETRAINED_INCEPTION_PATH) -> InceptionModel:
    "The trained model on DEVICE, in eval mode."
    return InceptionModel(inception_specs=weights).to(DEVICE).eval()


def preprocess(image: Image) -> torch.Tensor:
    "Any PIL image (RGBA, greyscale...) -> (3, 299, 299) normalized tensor, as FaceFolder does."
    return TRANSFORM_COMPOSITION(image.convert("RGB"))


def classify(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    "The one inference forward pass. x: (N, 3, H, W) -> (N, classes) softmax probabilities on CPU."
    device = next(model.parameters()).device
    with torch.no_grad():
        return model(x.to(device)).softmax(dim=1).cpu()


@dataclass(frozen=True)
class Prediction:
    x: torch.Tensor  # (1, 3, H, W) on the model's device, ready for saliency_map
    probabilities: torch.Tensor  # (classes,)
    class_names: list[str]

    @property
    def target(self) -> int:
        return int(self.probabilities.argmax())

    @property
    def label(self) -> str:
        return self.class_names[self.target]

    def by_class(self) -> dict[str, float]:
        return {n: float(p) for n, p in zip(self.class_names, self.probabilities)}


def predict(model: torch.nn.Module, x: torch.Tensor) -> Prediction:
    "One (3, H, W) normalized image -> its Prediction. `model` must have `class_names`."
    x = x.unsqueeze(0).to(next(model.parameters()).device)
    return Prediction(x, classify(model, x)[0], model.class_names)
