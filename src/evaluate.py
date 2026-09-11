"""Metrics (macro/micro-F1, AUC-PR, recall@K) and result bookkeeping."""

import json

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score

from src import ROOT

RESULTS = ROOT / "results"


def multilabel_metrics(probs: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> dict:
    """probs and targets are (n_clips, n_tags). AUC-PR skips tags with no positives in this split."""
    preds = probs >= threshold
    has_positive = targets.sum(axis=0) > 0
    return {
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(targets, preds, average="micro", zero_division=0)),
        "auc_pr": float(average_precision_score(targets[:, has_positive], probs[:, has_positive], average="macro")),
        "threshold": threshold,
    }


def best_threshold(probs: np.ndarray, targets: np.ndarray) -> float:
    """The single decision threshold that maximizes macro-F1 (pick it on validation data)."""
    grid = np.round(np.arange(0.05, 0.951, 0.05), 2)
    scores = [f1_score(targets, probs >= t, average="macro", zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def multiclass_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    """probs is (n_clips, n_classes); labels holds class indices."""
    preds = probs.argmax(axis=1)
    one_hot = np.eye(probs.shape[1])[labels]
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "auc_pr": float(average_precision_score(one_hot, probs, average="macro")),
    }


def retrieval_metrics(similarity: np.ndarray, ks: tuple[int, ...] = (1, 5, 10)) -> dict:
    """similarity[i, j] scores text i against clip j, where clip i is the true match for text i.

    mean_recall averages every R@K over both directions and is used for model selection.
    """
    scores = {}
    for direction, sim in (("text_to_audio", similarity), ("audio_to_text", similarity.T)):
        ranks = (sim > np.diag(sim)[:, None]).sum(axis=1)  # items scored above the true match
        for k in ks:
            scores[f"{direction}_R@{k}"] = float((ranks < k).mean())
        scores[f"{direction}_median_rank"] = float(np.median(ranks) + 1)
    scores["mean_recall"] = float(np.mean([v for key, v in scores.items() if "_R@" in key]))
    return scores


def save_metrics(name: str, metrics: dict) -> None:
    """Merge one experiment's metrics into results/metrics.json."""
    path = RESULTS / "metrics.json"
    everything = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    everything[name] = metrics
    path.write_text(json.dumps(everything, indent=2), encoding="utf-8")


def save_history(name: str, history: list[dict]) -> None:
    """Per-epoch records, used later to plot training curves."""
    path = RESULTS / "history" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=1), encoding="utf-8")
