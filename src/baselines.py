"""Baselines: random / majority predictors and a CNN on log-mel spectrograms."""

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score
from torch import nn

from src.evaluate import multiclass_metrics


def random_tag_baseline(train_targets: np.ndarray, test_targets: np.ndarray, seed: int) -> dict:
    """B1: switch each tag on at random with its training-set frequency; rank clips randomly for AUC-PR."""
    rng = np.random.default_rng(seed)
    guesses = rng.random(test_targets.shape) < train_targets.mean(axis=0)
    scores = rng.random(test_targets.shape)
    has_positive = test_targets.sum(axis=0) > 0
    return {
        "macro_f1": float(f1_score(test_targets, guesses, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(test_targets, guesses, average="micro", zero_division=0)),
        "auc_pr": float(average_precision_score(test_targets[:, has_positive], scores[:, has_positive], average="macro")),
    }


def majority_class_baseline(train_labels: np.ndarray, test_labels: np.ndarray, n_classes: int) -> dict:
    """B1 for single-label genre: always predict the most common training class."""
    majority = np.bincount(train_labels, minlength=n_classes).argmax()
    probs = np.zeros((len(test_labels), n_classes))
    probs[:, majority] = 1.0
    return multiclass_metrics(probs, test_labels)


class MelCNN(nn.Module):
    """B2: a VGG-style 2-D CNN over log-mel spectrograms, with no graph and no text."""

    def __init__(self, n_classes: int, channels: tuple[int, ...] = (32, 64, 128, 256), dropout: float = 0.3):
        super().__init__()
        blocks, c_in = [], 1
        for c_out in channels:
            blocks += [
                nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            c_in = c_out
        self.features = nn.Sequential(*blocks)
        self.dim = 2 * c_in
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.dim, n_classes))

    def embed(self, mel: torch.Tensor) -> torch.Tensor:
        """(batch, n_mels, frames) -> (batch, 2 * channels) via global mean and max pooling."""
        x = self.features(mel.unsqueeze(1))
        return torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(mel))


class MelDataset(torch.utils.data.Dataset):
    """Rows of a (possibly memory-mapped) float16 spectrogram array, paired with integer labels.

    With `crop`, each item is a random window of that many frames (training-time augmentation).
    """

    def __init__(self, mel: np.ndarray, rows: list[int], labels: list[int], crop: int | None = None):
        self.mel, self.rows, self.labels, self.crop = mel, rows, labels, crop

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        spectrogram = self.mel[self.rows[i]]
        if self.crop:
            start = np.random.randint(0, spectrogram.shape[1] - self.crop + 1)
            spectrogram = spectrogram[:, start : start + self.crop]
        return torch.from_numpy(np.asarray(spectrogram, dtype=np.float32)), self.labels[i]
