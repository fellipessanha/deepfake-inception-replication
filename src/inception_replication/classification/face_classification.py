import torch

from PIL.Image import Image
from torchvision import models
import os

from inception_replication.configs import PRETRAINED_INCEPTION_PATH, DEVICE


class InceptionModel(torch.nn.Module):
    def __init__(
        self,
        num_classes: int = 7,
        inception_specs: os.PathLike = PRETRAINED_INCEPTION_PATH,
    ) -> None:
        super().__init__()
        # No ImageNet weights: the checkpoint below overwrites every parameter anyway.
        self.inception_model: models.Inception3 = models.inception_v3(
            weights=None, aux_logits=False, init_weights=False
        )
        self.inception_model.fc = torch.nn.Linear(
            self.inception_model.fc.in_features, num_classes
        )
        _ = self.load_state_dict(
            torch.load(inception_specs, map_location=DEVICE, weights_only=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inception_model(x)


def classify_face(face: Image): ...
