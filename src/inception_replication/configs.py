import torch
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_CUTOFF_PROBABILITY: float = 0.8

ROOT_PATH = Path(__file__).parent.parent.parent
PRETRAINED_INCEPTION_PATH = (
    ROOT_PATH / "resources" / "best_inceptionv3_model_multiclass.h5"
)
