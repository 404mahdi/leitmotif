"""Extract audio features and build graphs for every clip, using most CPU cores.

    uv run python -m src.preprocess musiccaps
    uv run python -m src.preprocess fma_small

Writes to data/processed/<dataset>/:
    segment_graphs.pt, chord_graphs.pt   {clip id: torch_geometric Data}
    mel.npy, mel_ids.json                fixed-length log-mel spectrograms (float16) for the CNN baseline;
                                         row i belongs to mel_ids[i], and rows past len(mel_ids) are unused
    failed.json                          clips that could not be decoded, with the error
"""

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np
import torch

from src.audio_features import extract_features, load_audio
from src.datasets import load_fma_small, load_musiccaps
from src.graph_builder import chord_graph, segment_graph
from src.utils import load_config, resolve

_CFG: dict = {}


def _init_worker(cfg: dict) -> None:
    _CFG.update(cfg)
    torch.set_num_threads(1)


def fixed_length(x: np.ndarray, n_frames: int) -> np.ndarray:
    """Crop or zero-pad the time axis of a (n_bins, T) array to n_frames."""
    out = np.zeros((x.shape[0], n_frames), dtype=x.dtype)
    n = min(n_frames, x.shape[1])
    out[:, :n] = x[:, :n]
    return out


def n_frames_for(seconds: int, audio_cfg: dict) -> int:
    return 1 + seconds * audio_cfg["sample_rate"] // audio_cfg["hop_length"]


def process(item: tuple):
    key, path, seconds = item
    audio_cfg = _CFG["audio"]
    try:
        y = load_audio(path, audio_cfg["sample_rate"], max_seconds=seconds)
        if len(y) < 2 * audio_cfg["sample_rate"]:
            raise ValueError(f"only {len(y) / audio_cfg['sample_rate']:.1f} s of audio")
        features = extract_features(y, audio_cfg)
        return (
            key,
            segment_graph(features, _CFG["segments"], _CFG["graph"]),
            chord_graph(features, _CFG["segments"]),
            fixed_length(features.log_mel, n_frames_for(seconds, audio_cfg)).astype(np.float16),
            None,
        )
    except Exception as err:  # corrupt or unreadable files are logged and skipped
        return key, None, None, None, repr(err)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=["musiccaps", "fma_small"])
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    parser.add_argument("--limit", type=int, help="only process the first N clips (for testing)")
    args = parser.parse_args()

    cfg = load_config()
    if args.dataset == "musiccaps":
        df = load_musiccaps(cfg)
        df = df[df["has_audio"]]
        seconds = cfg["data"]["musiccaps"]["clip_seconds"]
        items = [(ytid, str(path), seconds) for ytid, path in zip(df["ytid"], df["audio_path"])]
    else:
        df = load_fma_small(cfg)
        seconds = cfg["data"]["fma"]["clip_seconds"]
        items = [(int(track), str(path), seconds) for track, path in zip(df["track_id"], df["audio_path"])]
    items = items[: args.limit]

    out = resolve(cfg["paths"]["processed"]) / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    mel = np.lib.format.open_memmap(
        out / "mel.npy", mode="w+", dtype=np.float16,
        shape=(len(items), cfg["audio"]["n_mels"], n_frames_for(seconds, cfg["audio"])),
    )
    segment, chord, ids, failed = {}, {}, [], {}
    started = time.time()
    print(f"Processing {len(items)} clips with {args.workers} workers", flush=True)
    with mp.Pool(args.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        for i, (key, seg, chords, spectrogram, error) in enumerate(pool.imap_unordered(process, items, chunksize=4), 1):
            if error:
                failed[str(key)] = error
            else:
                mel[len(ids)] = spectrogram
                ids.append(key)
                segment[key], chord[key] = seg, chords
            if i % 250 == 0 or i == len(items):
                print(f"{i}/{len(items)} | failed {len(failed)} | {i / (time.time() - started):.1f} clips/s", flush=True)
    mel.flush()
    del mel

    torch.save(segment, out / "segment_graphs.pt")
    torch.save(chord, out / "chord_graphs.pt")
    (out / "mel_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    (out / "failed.json").write_text(json.dumps(failed, indent=1), encoding="utf-8")
    print(f"Saved {len(ids)} clips to {out.relative_to(resolve('.'))}; {len(failed)} failed")


if __name__ == "__main__":
    main()
