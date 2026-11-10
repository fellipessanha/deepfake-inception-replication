import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PIL import Image

from inception_replication.classification.inception_model import load_model, preprocess
from inception_replication.saliency.explainers import ExplainerMethods
from inception_replication.saliency.spotlight import spotlight
from inception_replication.saliency.visualize import explain

GRADCAM = ExplainerMethods.GradCAM


def vlm_input(model, image: Image.Image) -> tuple[np.ndarray, np.ndarray, dict]:
    """-> GradCAM spotlight, GradCAM heatmap overlay, prediction record.

    Both images are (H, W, 3) in [0, 1] at the image's own resolution, not the model's 299x299.
    """
    image = image.convert("RGB")
    pred, maps = explain(model, preprocess(image), (GRADCAM,))
    # stretch the 299x299 map back over the original, so the VLM sees full texture detail
    cam = np.asarray(
        Image.fromarray(maps[GRADCAM.name].astype(np.float32)).resize(
            image.size, Image.Resampling.BILINEAR
        )
    )
    raw = np.asarray(image, dtype=np.float32) / 255
    lit, box = spotlight(raw, cam)
    overlay = 0.5 * raw + 0.5 * colormaps["jet"](cam)[..., :3]
    record = {
        "predicted": pred.label,
        "probabilities": {n: round(p, 4) for n, p in pred.by_class().items()},
        "gradcam_box": box,
        "box_format": "[x1, y1, x2, y2], each 0-1000 relative to the image size",
    }
    return lit, overlay, record


def _save(array: np.ndarray, path: Path) -> None:
    Image.fromarray((array.clip(0, 1) * 255).astype(np.uint8)).save(path)


def _main() -> None:
    # shortcut: one image per call; loop in the shell or add a folder mode for batches
    parser = argparse.ArgumentParser(
        description="Write the GradCAM images and class probabilities a VLM needs for one image."
    )
    parser.add_argument("path", help="a single face image")
    parser.add_argument("-o", "--output-dir", default="vlm_input")
    args = parser.parse_args()

    lit, overlay, record = vlm_input(load_model(), Image.open(args.path))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.path).stem
    _save(lit, out / f"{stem}_spotlight.png")
    _save(overlay, out / f"{stem}_gradcam.png")
    record["image"] = str(args.path)
    (out / f"{stem}.json").write_text(json.dumps(record, indent=2))
    print(f"{record['predicted']}: written to {out}/{stem}_{{spotlight,gradcam}}.png and {stem}.json")


if __name__ == "__main__":
    _main()
