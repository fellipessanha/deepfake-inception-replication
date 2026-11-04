import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_CUTOFF_PROBABILITY: float = 0.8
