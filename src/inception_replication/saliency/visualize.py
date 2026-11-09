import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, no window needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from PIL import Image

from inception_replication.classification.face_folder import (
    IMAGENET_MEAN_PARAMS,
    IMAGENET_STD_PARAMS,
    TRANSFORM_COMPOSITION,
    FaceFolder,
)
from inception_replication.classification.inception_model import InceptionModel
from inception_replication.configs import DEVICE
from inception_replication.saliency.explainers import ExplainerMethods, saliency_map
from inception_replication.saliency.spotlight import spotlight


def to_raw_image(x: torch.Tensor) -> torch.Tensor:
    "Undo Normalize: (3, H, W) normalized tensor -> (H, W, 3) in [0, 1] for imshow."
    mean = torch.tensor(IMAGENET_MEAN_PARAMS)
    std = torch.tensor(IMAGENET_STD_PARAMS)
    return (x.cpu().permute(1, 2, 0) * std + mean).clamp(0, 1)


def explain(
    model: torch.nn.Module, x: torch.Tensor, methods=tuple(ExplainerMethods)
) -> tuple[int, dict[str, np.ndarray]]:
    "x: (3, H, W) normalized image -> predicted class index, {method name: saliency map}."
    x = x.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        target = int(model(x).argmax(dim=1).item())
    return target, {m.name: saliency_map(m, model, x, target) for m in methods}


def plot_explanation(
    raw: torch.Tensor, maps: dict[str, np.ndarray], title: str, axes=None
) -> np.ndarray:
    "One row: the photo, then one overlay per map. Draws on `axes` if given, else a new figure."
    if axes is None:
        _, axes = plt.subplots(1, len(maps) + 1, figsize=(3 * (len(maps) + 1), 3))
    photo: Axes = axes[0]
    photo.imshow(raw)
    photo.set_title(title, fontsize=9)
    for ax, (name, saliency) in zip(axes[1:], maps.items()):
        ax.imshow(raw)
        ax.imshow(saliency, cmap="jet", alpha=0.5)
        ax.set_title(name, fontsize=9)
    for ax in axes:
        ax.axis("off")
    return axes


def plot_spotlight(ax: Axes, raw: torch.Tensor, saliency: np.ndarray, title: str) -> None:
    "The spotlight image, with its box converted back from Qwen 0-1000 coords to pixels."
    lit, (x1, y1, x2, y2) = spotlight(np.asarray(raw), saliency)
    h, w = saliency.shape
    ax.imshow(lit)
    ax.add_patch(
        Rectangle(
            (x1 * w / 1000, y1 * h / 1000),
            (x2 - x1) * w / 1000,
            (y2 - y1) * h / 1000,
            fill=False,
            edgecolor="lime",
            linewidth=1,
        )
    )
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def plot_explanation_with_spotlight(
    raw: torch.Tensor, maps: dict[str, np.ndarray], title: str, axes=None
) -> np.ndarray:
    "plot_explanation's row plus one last column: the GradCAM spotlight. `maps` must hold GradCAM."
    if axes is None:
        columns = len(maps) + 2
        _, axes = plt.subplots(1, columns, figsize=(3 * columns, 3))
    plot_explanation(raw, maps, title, axes=axes[:-1])
    name = ExplainerMethods.GradCAM.name
    plot_spotlight(axes[-1], raw, maps[name], f"{name} spotlight")
    return axes


def saliency_grid(model, dataset: FaceFolder, indices, methods=tuple(ExplainerMethods)):
    "dados cell 11: one plot_explanation row per dataset image."
    columns = len(methods) + 1
    fig, rows = plt.subplots(
        len(indices), columns, figsize=(3 * columns, 3 * len(indices)), squeeze=False
    )
    names = dataset.class_names
    for row, idx in zip(rows, indices):
        x, label = dataset[idx]
        target, maps = explain(model, x, methods)
        title = f"real: {names[label]}\npred: {names[target]}"
        plot_explanation(to_raw_image(x), maps, title, axes=row)
    fig.tight_layout()
    return fig


def explain_image(model, image_path, methods=tuple(ExplainerMethods)):
    "A single face image from disk: same preprocessing as FaceFolder, no true label."
    x = TRANSFORM_COMPOSITION(Image.open(image_path).convert("RGB"))
    target, maps = explain(model, x, methods)
    axes = plot_explanation(to_raw_image(x), maps, f"pred: {model.class_names[target]}")
    fig = axes[0].figure
    fig.tight_layout()
    return fig


def _main() -> None:
    parser = argparse.ArgumentParser(description="Plot saliency maps for face images.")
    parser.add_argument(
        "path", help="a single face image, or an ImageFolder root (random sample)"
    )
    parser.add_argument("-o", "--output", default="saliency_grid.png")
    parser.add_argument("-n", "--count", type=int, default=4, help="images, for a folder")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = InceptionModel().to(DEVICE).eval()
    if Path(args.path).is_file():
        fig = explain_image(model, args.path)
    else:
        dataset = FaceFolder(args.path)
        generator = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(len(dataset), generator=generator)[: args.count]
        fig = saliency_grid(model, dataset, indices.tolist())

    fig.savefig(args.output, dpi=100)
    print(f"saliency plot written to {args.output}")


def _spotlight_main() -> None:
    # shortcut: single image only, since saliency_grid has no spotlight column; add a folder mode if needed
    parser = argparse.ArgumentParser(
        description="Plot saliency maps plus the GradCAM spotlight for one face image."
    )
    parser.add_argument("path", help="a single face image")
    parser.add_argument("-o", "--output", default="saliency_spotlight.png")
    args = parser.parse_args()

    model = InceptionModel().to(DEVICE).eval()
    x = TRANSFORM_COMPOSITION(Image.open(args.path).convert("RGB"))
    target, maps = explain(model, x)
    title = f"pred: {model.class_names[target]}"
    axes = plot_explanation_with_spotlight(to_raw_image(x), maps, title)
    fig = axes[0].figure
    fig.tight_layout()
    fig.savefig(args.output, dpi=100)
    print(f"spotlight plot written to {args.output}")


if __name__ == "__main__":
    _main()
