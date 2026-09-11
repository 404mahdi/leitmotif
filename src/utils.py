"""Config loading and reproducibility helpers."""

import random
from pathlib import Path

import numpy as np
import torch
import yaml

from src import ROOT


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Turn a repo-relative path from config.yaml into an absolute one."""
    return ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
