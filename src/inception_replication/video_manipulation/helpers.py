from collections.abc import Iterator
from pathlib import Path

import cv2
from facenet_pytorch import MTCNN
from numpy.typing import ArrayLike
from PIL import Image

from inception_replication.configs import DEFAULT_CUTOFF_PROBABILITY, DEVICE


def iterate_frames(capture: Path | str | cv2.VideoCapture) -> Iterator[ArrayLike]:
    if not isinstance(capture, cv2.VideoCapture):
        capture = cv2.VideoCapture(capture)

    ok, frame = capture.read()
    while ok:
        yield frame
        ok, frame = capture.read()
    capture.release()


def _crop_faces(
    image: Image.Image,
    detect_output: tuple[ArrayLike, ArrayLike],
    cutoff_probability: float = DEFAULT_CUTOFF_PROBABILITY,
) -> None | Image.Image:
    box, probability = detect_output
    if probability < cutoff_probability:
        return None
    x1, y1, x2, y2 = box.round().astype(int)
    return image.crop((x1, y1, x2, y2))


def _detect_faces(frame: ArrayLike) -> list[Image.Image]:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detector = MTCNN(keep_all=True, device=DEVICE)
    boxes, probabilities = detector.detect(image)
    if boxes is None:
        return []

    faces = [_crop_faces(image, t) for t in zip(boxes, probabilities)]
    return [face for face in faces if face is not None]
