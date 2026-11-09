import numpy as np
from PIL import Image, ImageFilter


def spotlight(
    raw: np.ndarray, saliency: np.ndarray, keep=0.15, dim=0.25, feather=4
) -> tuple[np.ndarray, list[int]]:
    """raw: (H, W, 3) in [0, 1]; saliency: (H, W) in [0, 1].

    Keeps the top `keep` fraction of pixels in original colour, dims and greys the rest.
    Returns the image and the lit region's box [x1, y1, x2, y2] in Qwen-VL 0-1000 coords.
    """
    hard = saliency >= np.quantile(saliency, 1 - keep)
    # feathered edge: a hard cut-out edge itself looks like a splice artifact to the VLM
    soft = Image.fromarray(hard.astype(np.uint8) * 255).filter(
        ImageFilter.GaussianBlur(feather)
    )
    mask = np.asarray(soft, dtype=np.float32)[..., None] / 255
    grey = raw.mean(axis=2, keepdims=True).repeat(3, axis=2)
    out = raw * mask + grey * dim * (1 - mask)
    ys, xs = np.nonzero(hard)
    h, w = saliency.shape
    box = [
        int(v)
        for v in (
            xs.min() / w * 1000,
            ys.min() / h * 1000,
            xs.max() / w * 1000,
            ys.max() / h * 1000,
        )
    ]
    return out, box
