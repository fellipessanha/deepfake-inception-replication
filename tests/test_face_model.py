import pytest
import torch
from PIL import Image
from torchvision.datasets.folder import default_loader

from inception_replication.classification.face_folder import (
    IMAGENET_MEAN_PARAMS,
    IMAGENET_STD_PARAMS,
    FaceFolder,
)
from inception_replication.configs import CLASS_NAMES


@pytest.fixture(autouse=True)
def no_disk(monkeypatch):
    # Disk scanning is torchvision's job; fake it so only our class selection is under test.
    def fake_make_dataset(directory, class_to_idx, **_):
        return [(f"{directory}/{name}/img.png", idx) for name, idx in class_to_idx.items()]

    monkeypatch.setattr(FaceFolder, "make_dataset", staticmethod(fake_make_dataset))


def test_uses_model_classes_in_model_order():
    ds = FaceFolder("any/root")
    assert ds.classes == CLASS_NAMES
    assert ds.class_to_idx == {name: i for i, name in enumerate(CLASS_NAMES)}


def test_labels_match_class_names():
    ds = FaceFolder("any/root")
    for path, label in ds.samples:
        assert f"/{CLASS_NAMES[label]}/" in path


def test_custom_classes():
    ds = FaceFolder("any/root", valid_classes=["b", "a"])
    assert ds.class_to_idx == {"b": 0, "a": 1}  # given order kept, not re-sorted


def test_default_loader_is_torchvision_default():
    assert FaceFolder("any/root").loader is default_loader


def _dataset_of(image, **kwargs):
    # In-memory loader: every sample "opens" to the same image, no file read.
    return FaceFolder("any/root", loader=lambda _: image, **kwargs)


def test_default_transform_stretches_to_inception_input():
    x, _ = _dataset_of(Image.new("RGB", (100, 60)))[0]  # non-square, like a face crop
    assert x.shape == (3, 299, 299)
    assert x.dtype == torch.float32


def test_default_transform_normalizes_with_imagenet_stats():
    x, _ = _dataset_of(Image.new("RGB", (8, 8), (128, 128, 128)))[0]
    expected = (128 / 255 - torch.tensor(IMAGENET_MEAN_PARAMS)) / torch.tensor(IMAGENET_STD_PARAMS)
    assert torch.allclose(x.mean(dim=(1, 2)), expected, atol=1e-5)


def test_transform_can_be_overridden():
    image = Image.new("RGB", (8, 8))
    x, _ = _dataset_of(image, transform=None)[0]
    assert x is image
