"""Save a handful of preprocessed graphs as .pt and .json, for the submission and for quick inspection.

    uv run python -m src.export_samples      # 10 FMA-small + 10 MusicCaps test clips

For each clip, data/processed/samples/<dataset>_<id>.pt holds its segment and chord graphs as
PyG Data objects, and the .json twin holds the same graphs as plain lists plus the clip's labels.
"""

import json

import numpy as np
import torch

from src.datasets import load_fma_small, load_musiccaps
from src.utils import load_config, resolve

NODE_FEATURES = "segment graph x: log-mel mean (128) + chroma mean (12) + MFCC mean (20) + MFCC std (20); " \
    "chord graph x: chord one-hot (24 triads + N) + time share (1) + the same 180 audio features averaged"
EDGE_COLUMNS = {"segment_graph": ["is_temporal", "is_similar", "cosine_similarity"], "chord_graph": ["transition_count"]}


def rounded(tensor: torch.Tensor) -> list:
    return np.round(tensor.numpy().astype(float), 4).tolist()


def graph_json(data, kind: str) -> dict:
    record = {
        "num_nodes": int(data.num_nodes),
        "edge_index": data.edge_index.tolist(),
        "edge_attr_columns": EDGE_COLUMNS[kind],
        "edge_attr": rounded(data.edge_attr),
        "x": rounded(data.x),
    }
    if "segment_seconds" in data:
        record["segment_seconds"] = rounded(data.segment_seconds)
    if "chords" in data:
        record["chords"] = list(data.chords)
    return record


def main() -> None:
    cfg = load_config()
    out = resolve(cfg["paths"]["processed"]) / "samples"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])

    datasets = (
        ("fma_small", load_fma_small(cfg), "track_id", lambda row: {"genre": row["genre"]}),
        (
            "musiccaps", load_musiccaps(cfg), "ytid",
            lambda row: {
                "caption": row["caption"],
                "tags": row["tags"],
                "youtube": f"https://www.youtube.com/watch?v={row['ytid']}&t={int(row['start_s'])}s",
            },
        ),
    )
    for dataset, frame, id_column, describe in datasets:
        root = resolve(cfg["paths"]["processed"]) / dataset
        if not (root / "segment_graphs.pt").exists():
            print(f"Skipping {dataset}: run `python -m src.preprocess {dataset}` first")
            continue
        graphs = {kind: torch.load(root / f"{kind}s.pt", weights_only=False) for kind in EDGE_COLUMNS}
        rows = frame[frame[id_column].isin(list(graphs["segment_graph"])) & (frame["split"] == "test")]
        picked = rows.iloc[sorted(rng.choice(len(rows), size=min(10, len(rows)), replace=False))]
        for _, row in picked.iterrows():
            key = row[id_column]
            name = f"{dataset}_{key}"
            torch.save({kind: graphs[kind][key] for kind in EDGE_COLUMNS}, out / f"{name}.pt")
            record = {
                "dataset": dataset,
                "id": key if isinstance(key, str) else int(key),
                "split": row["split"],
                **describe(row),
                "node_features": NODE_FEATURES,
                **{kind: graph_json(graphs[kind][key], kind) for kind in EDGE_COLUMNS},
            }
            (out / f"{name}.json").write_text(json.dumps(record), encoding="utf-8")
        print(f"{dataset}: exported {len(picked)} clips to {out.relative_to(resolve('.'))}")


if __name__ == "__main__":
    main()
