"""Audio preprocessing: load at 22,050 Hz, extract log-mel, chroma and MFCC
features, normalize per track, and cut into fixed or beat-synchronous segments."""

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np


@dataclass
class Features:
    """Frame-level features of one clip, each shaped (n_features, n_frames)."""

    log_mel: np.ndarray  # z-scored over the whole clip
    chroma: np.ndarray  # CQT chroma, each frame scaled to a max of 1
    mfcc: np.ndarray
    sample_rate: int
    hop_length: int

    @property
    def n_frames(self) -> int:
        return self.log_mel.shape[1]

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop_length


def load_audio(path: str | Path, sample_rate: int, max_seconds: float | None = None) -> np.ndarray:
    y, _ = librosa.load(path, sr=sample_rate, mono=True, duration=max_seconds)
    return y


def extract_features(y: np.ndarray, audio_cfg: dict) -> Features:
    sr, hop = audio_cfg["sample_rate"], audio_cfg["hop_length"]
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=audio_cfg["n_fft"], hop_length=hop, n_mels=audio_cfg["n_mels"]
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop, n_chroma=audio_cfg["n_chroma"])
    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=audio_cfg["n_mfcc"])
    n = min(mel_db.shape[1], chroma.shape[1])  # CQT and STFT framing can differ by a frame
    log_mel = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    return Features(
        log_mel=log_mel[:, :n].astype(np.float32),
        chroma=chroma[:, :n].astype(np.float32),
        mfcc=mfcc[:, :n].astype(np.float32),
        sample_rate=sr,
        hop_length=hop,
    )


def segment_bounds(features: Features, seg_cfg: dict) -> np.ndarray:
    """Start and end frame of each segment, shaped (n_segments, 2)."""
    n = features.n_frames
    if seg_cfg["mode"] == "beat":
        onset = librosa.onset.onset_strength(
            S=features.log_mel, sr=features.sample_rate, hop_length=features.hop_length
        )
        _, beats = librosa.beat.beat_track(
            onset_envelope=onset, sr=features.sample_rate, hop_length=features.hop_length
        )
        edges = np.unique(np.concatenate([[0], beats, [n]]))
        bounds = np.stack([edges[:-1], edges[1:]], axis=1)
        return bounds[bounds[:, 1] - bounds[:, 0] >= 2]  # drop slivers shorter than two frames
    win, hop = window_and_hop(seg_cfg, features.frame_rate)
    starts = np.arange(0, max(n - win, 0) + 1, hop)
    return np.stack([starts, np.minimum(starts + win, n)], axis=1)


def window_and_hop(seg_cfg: dict, frame_rate: float) -> tuple[int, int]:
    """Length and hop of fixed segments, in frames."""
    return max(1, round(seg_cfg["window_seconds"] * frame_rate)), max(1, round(seg_cfg["hop_seconds"] * frame_rate))


def segment_patches(log_mel: np.ndarray, n_segments: int, window: int, hop: int) -> np.ndarray:
    """Cut a (n_mels, T) spectrogram into fixed-window segment patches shaped (n_segments, n_mels, window).

    A window running past the end repeats the last frame, which only happens for clips shorter than one window.
    """
    frames = np.minimum(np.arange(n_segments)[:, None] * hop + np.arange(window), log_mel.shape[1] - 1)
    return np.ascontiguousarray(log_mel[:, frames].transpose(1, 0, 2))


def segment_features(features: Features, bounds: np.ndarray) -> dict[str, np.ndarray]:
    """Pool the frames inside each segment into arrays shaped (n_segments, dim)."""

    def pool(x: np.ndarray, fn) -> np.ndarray:
        return np.stack([fn(x[:, start:end], axis=1) for start, end in bounds])

    return {
        "log_mel": pool(features.log_mel, np.mean),
        "chroma": pool(features.chroma, np.mean),
        "mfcc_mean": pool(features.mfcc, np.mean),
        "mfcc_std": pool(features.mfcc, np.std),
    }
