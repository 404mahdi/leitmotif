"""Music structure graphs as PyTorch Geometric `Data` objects.

- Segment graph: nodes are time segments; edges join neighbors in time and
  segments whose chroma/MFCC cosine similarity is above tau.
- Chord graph: nodes are chords estimated from chroma; edges are observed
  transitions weighted by count.
"""

import numpy as np
import torch
from torch_geometric.data import Data

from src.audio_features import Features, segment_bounds, segment_features

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CHORDS = [f"{note}:{quality}" for quality in ("maj", "min") for note in NOTES]
NO_CHORD = "N"


def _chord_templates() -> np.ndarray:
    """Unit-norm 12-bin templates for the 24 major and minor triads, in CHORDS order."""
    templates = []
    for third in (4, 3):  # major, minor
        for root in range(12):
            template = np.zeros(12)
            template[[root, (root + third) % 12, (root + 7) % 12]] = 1
            templates.append(template / np.linalg.norm(template))
    return np.array(templates)


TEMPLATES = _chord_templates()


def node_features(pooled: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate pooled log-mel, chroma and MFCC stats: 128 + 12 + 20 + 20 = 180 dims."""
    return np.concatenate([pooled["log_mel"], pooled["chroma"], pooled["mfcc_mean"], pooled["mfcc_std"]], axis=1)


def segment_graph(features: Features, seg_cfg: dict, graph_cfg: dict) -> Data:
    bounds = segment_bounds(features, seg_cfg)
    pooled = segment_features(features, bounds)
    n = len(bounds)
    idx = np.arange(n)

    temporal = np.zeros((n, n), dtype=bool)
    temporal[idx[:-1], idx[1:]] = temporal[idx[1:], idx[:-1]] = True

    # Compare segments after removing the clip's average sound, so similarity
    # reflects what makes two moments alike rather than what the whole clip shares
    key = "chroma" if graph_cfg["similarity_feature"] == "chroma" else "mfcc_mean"
    centered = pooled[key] - pooled[key].mean(axis=0)
    unit = centered / (np.linalg.norm(centered, axis=1, keepdims=True) + 1e-8)
    cosine = unit @ unit.T

    candidates = cosine.copy()
    candidates[np.abs(idx[:, None] - idx[None, :]) <= 1] = -np.inf  # self and overlapping windows
    k = min(graph_cfg["max_similar_neighbors"], n)
    rows = np.repeat(idx, k)
    cols = np.argsort(-candidates, axis=1)[:, :k].ravel()
    keep = candidates[rows, cols] > graph_cfg["similarity_threshold"]
    similar = np.zeros((n, n), dtype=bool)
    similar[rows[keep], cols[keep]] = True
    similar |= similar.T

    src, dst = np.nonzero(temporal | similar)
    edge_attr = np.stack([temporal[src, dst], similar[src, dst], cosine[src, dst]], axis=1)
    return Data(
        x=torch.from_numpy(node_features(pooled)).float(),
        edge_index=torch.from_numpy(np.stack([src, dst])).long(),
        edge_attr=torch.from_numpy(edge_attr).float(),  # [is_temporal, is_similar, cosine]
        segment_seconds=torch.from_numpy(bounds / features.frame_rate).float(),
    )


def estimate_chords(chroma: np.ndarray, min_confidence: float = 0.6) -> list[str]:
    """Label each row of pooled chroma with its best-matching triad, or N when none fits.

    A flat chroma vector scores 0.5 against every triad, so the threshold keeps
    noise and drums from being labelled as chords.
    """
    unit = chroma / (np.linalg.norm(chroma, axis=1, keepdims=True) + 1e-8)
    scores = unit @ TEMPLATES.T
    best = scores.argmax(axis=1)
    return [CHORDS[b] if scores[i, b] >= min_confidence else NO_CHORD for i, b in enumerate(best)]


def chord_graph(features: Features, seg_cfg: dict) -> Data:
    """Nodes are the distinct chords in a clip; directed edges count chord changes."""
    bounds = segment_bounds(features, {**seg_cfg, "mode": "beat"})  # chords change on beats
    pooled = segment_features(features, bounds)
    labels = estimate_chords(pooled["chroma"])
    names = list(dict.fromkeys(labels))
    node_of = {name: i for i, name in enumerate(names)}

    counts = np.zeros((len(names), len(names)), dtype=np.float32)
    for a, b in zip(labels, labels[1:]):
        if a != b:
            counts[node_of[a], node_of[b]] += 1
    src, dst = np.nonzero(counts)

    frames = bounds[:, 1] - bounds[:, 0]
    per_segment = node_features(pooled)
    x = []
    for name in names:
        members = [i for i, label in enumerate(labels) if label == name]
        identity = np.zeros(len(CHORDS) + 1)
        identity[CHORDS.index(name) if name != NO_CHORD else len(CHORDS)] = 1
        share = frames[members].sum() / frames.sum()
        x.append(np.concatenate([identity, [share], per_segment[members].mean(axis=0)]))

    return Data(
        x=torch.tensor(np.array(x)).float(),
        edge_index=torch.from_numpy(np.stack([src, dst])).long(),
        edge_attr=torch.from_numpy(counts[src, dst][:, None]),
        chords=names,
    )


def build_graph(features: Features, cfg: dict) -> Data:
    if cfg["graph"]["type"] == "chord":
        return chord_graph(features, cfg["segments"])
    return segment_graph(features, cfg["segments"], cfg["graph"])
