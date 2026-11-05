from collections.abc import Callable
from pathlib import Path
from typing import Any, override

from torchvision import datasets

from inception_replication.configs import CLASS_NAMES


class FaceModel(datasets.ImageFolder):
    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
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
