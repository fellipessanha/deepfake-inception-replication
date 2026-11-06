from collections.abc import Callable
from pathlib import Path
from typing import Any, override

from torchvision import datasets, transforms

from inception_replication.configs import CLASS_NAMES

IMAGENET_MEAN_PARAMS = [0.485, 0.456, 0.406]
IMAGENET_STD_PARAMS = [0.229, 0.224, 0.225]
TRANSFORM_COMPOSITION = transforms.Compose(
    [
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN_PARAMS, std=IMAGENET_STD_PARAMS),
    ]
)


class FaceModel(datasets.ImageFolder):
    def __init__(
        self,
        root: str | Path,
        transform=TRANSFORM_COMPOSITION,
        target_transform=None,
        loader: Callable[[str], Any] = datasets.folder.default_loader,
        is_valid_file: Callable[[str], bool] | None = None,
        allow_empty: bool = False,
        valid_classes: list[str] = CLASS_NAMES,
    ):
        self.class_names: list[str] = valid_classes
        super().__init__(
            root, transform, target_transform, loader, is_valid_file, allow_empty
        )

    @override
    def find_classes(self, directory: str | Path) -> tuple[list[str], dict[str, int]]:
        return self.class_names, {name: i for i, name in enumerate(self.class_names)}
