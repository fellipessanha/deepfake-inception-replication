import pytest
from torchvision.datasets.folder import default_loader

from inception_replication.classification.face_model import FaceModel
from inception_replication.configs import CLASS_NAMES


@pytest.fixture(autouse=True)
def no_disk(monkeypatch):
    # Disk scanning is torchvision's job; fake it so only our class selection is under test.
    def fake_make_dataset(directory, class_to_idx, **_):
        return [(f"{directory}/{name}/img.png", idx) for name, idx in class_to_idx.items()]

    monkeypatch.setattr(FaceModel, "make_dataset", staticmethod(fake_make_dataset))


def test_uses_model_classes_in_model_order():
    ds = FaceModel("any/root")
    assert ds.classes == CLASS_NAMES
    assert ds.class_to_idx == {name: i for i, name in enumerate(CLASS_NAMES)}


def test_labels_match_class_names():
    ds = FaceModel("any/root")
    for path, label in ds.samples:
        assert f"/{CLASS_NAMES[label]}/" in path


def test_custom_classes():
    ds = FaceModel("any/root", valid_classes=["b", "a"])
    assert ds.class_to_idx == {"b": 0, "a": 1}  # given order kept, not re-sorted


def test_default_loader_is_torchvision_default():
    assert FaceModel("any/root").loader is default_loader
