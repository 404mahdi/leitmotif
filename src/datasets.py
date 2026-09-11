"""Dataset loaders for MusicCaps and FMA, the tag vocabulary, and train/val/test splits.

Run `python -m src.datasets` to write the split and vocabulary files in data/splits/.
"""

import ast
import hashlib
import json
import re
from collections import Counter

import numpy as np
import pandas as pd

from src.tags import canonical_tags, phrases_for
from src.utils import load_config, resolve

FMA_SPLITS = {"training": "train", "validation": "val", "test": "test"}


def musiccaps_split(ytid: str, is_audioset_eval: bool) -> str:
    """MusicCaps has no official split, so the AudioSet eval clips are the test set and
    a stable hash of the video id sends 10% of the remaining clips to validation."""
    if is_audioset_eval:
        return "test"
    return "val" if int(hashlib.md5(ytid.encode()).hexdigest(), 16) % 10 == 0 else "train"


def load_musiccaps(cfg: dict) -> pd.DataFrame:
    root = resolve(cfg["paths"]["raw"]) / "musiccaps"
    df = pd.read_csv(root / "musiccaps-public.csv")
    df["aspects"] = df["aspect_list"].map(lambda s: [a.strip().lower() for a in ast.literal_eval(s)])
    df["tags"] = df["aspects"].map(canonical_tags)
    df["split"] = [musiccaps_split(y, e) for y, e in zip(df["ytid"], df["is_audioset_eval"])]
    df["audio_path"] = df["ytid"].map(lambda y: root / "audio" / f"{y}.wav")
    df["has_audio"] = df["audio_path"].map(lambda p: p.exists())
    return df


def tag_vocabulary(df: pd.DataFrame, k: int) -> list[str]:
    """The k most frequent canonical tags, counted on the training split only."""
    counts = Counter(tag for tags in df.loc[df["split"] == "train", "tags"] for tag in tags)
    return [tag for tag, _ in counts.most_common(k)]


def multi_hot(tags: list[str], vocab: list[str]) -> np.ndarray:
    index = {tag: i for i, tag in enumerate(vocab)}
    y = np.zeros(len(vocab), dtype=np.float32)
    y[[index[t] for t in tags if t in index]] = 1
    return y


def tag_mask_pattern(vocab: list[str]) -> re.Pattern:
    """Regex matching any phrase that maps to a vocabulary tag, longest phrases first."""
    phrases = sorted({p for tag in vocab for p in phrases_for(tag)}, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(p) for p in phrases) + r")\b", re.IGNORECASE)


def mask_tags(caption: str, pattern: re.Pattern, mask_token: str = "[MASK]") -> str:
    """Hide tag words so a text model can't just spot the label in the caption."""
    return pattern.sub(mask_token, caption)


def load_fma_small(cfg: dict) -> pd.DataFrame:
    """One row per FMA-small track with its top genre, official split and mp3 path."""
    root = resolve(cfg["paths"]["raw"]) / "fma"
    tracks = pd.read_csv(root / "fma_metadata" / "tracks.csv", index_col=0, header=[0, 1], low_memory=False)
    small = tracks[tracks["set", "subset"] == "small"]
    df = pd.DataFrame({
        "track_id": small.index,
        "artist_id": small["artist", "id"].to_numpy(),
        "genre": small["track", "genre_top"].astype(str).to_numpy(),
        "split": small["set", "split"].map(FMA_SPLITS).to_numpy(),
    })
    df["audio_path"] = df["track_id"].map(lambda t: root / "fma_small" / f"{t:06d}"[:3] / f"{t:06d}.mp3")
    return df


def write_splits(cfg: dict) -> None:
    out = resolve(cfg["paths"]["splits"])
    out.mkdir(parents=True, exist_ok=True)

    caps = load_musiccaps(cfg)
    vocab = tag_vocabulary(caps, cfg["data"]["musiccaps"]["top_k_tags"])
    splits = {name: sorted(caps.loc[caps["split"] == name, "ytid"]) for name in ("train", "val", "test")}
    (out / "musiccaps.json").write_text(json.dumps(splits, indent=1), encoding="utf-8")
    train_counts = Counter(t for tags in caps.loc[caps["split"] == "train", "tags"] for t in tags)
    (out / "musiccaps_tags.json").write_text(
        json.dumps({tag: train_counts[tag] for tag in vocab}, indent=1), encoding="utf-8"
    )

    fma = load_fma_small(cfg)
    fma_splits = {name: fma.loc[fma["split"] == name, "track_id"].tolist() for name in ("train", "val", "test")}
    (out / "fma_small.json").write_text(json.dumps(fma_splits, indent=1), encoding="utf-8")

    print("MusicCaps:", {k: len(v) for k, v in splits.items()}, f"| {len(vocab)} tags")
    print("FMA-small:", {k: len(v) for k, v in fma_splits.items()})


if __name__ == "__main__":
    write_splits(load_config())
