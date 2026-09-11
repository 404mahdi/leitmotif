"""Report and README figures, drawn from results/ and checkpoints/ with one fixed palette.

    uv run python -m src.plots

A figure is skipped when its inputs don't exist yet. Each one is written to
results/plots/ as PNG (for the README) and PDF (for the LaTeX report).
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from src import ROOT
from src.evaluate import RESULTS
from src.utils import load_config, resolve

PLOTS = RESULTS / "plots"
CHECKPOINTS = ROOT / "checkpoints"

# Light-surface palette. Categorical slots are always used in this order; slots 1-3 are
# validated for colorblind separation across all pairs, so scatter plots stop at three.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
BLUE = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)

GENRE_TAGS = ["rock", "pop", "electronic", "classical", "folk", "dance", "hip hop", "jazz", "metal", "country"]
MOOD_TAGS = ["energetic", "relaxing", "sad", "happy", "eerie", "romantic", "mellow"]
VARIANT_NAMES = {
    "gnn_only": "GNN only (audio graph)",
    "bert_only": "BERT only (masked caption)",
    "concat": "Early concat",
    "cross_attention": "Cross-attention",
}


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
        "text.color": INK, "axes.titlecolor": INK, "axes.labelcolor": INK_SECONDARY,
        "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "legend.fontsize": 8,
    })


def save(fig, name: str) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote results/plots/{name}.png and .pdf")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def pick(metrics: dict, *names: str):
    """The first of several experiment names that has results."""
    return next((metrics[name] for name in names if name in metrics), None)


def hbars(ax, labels: list[str], values: list[float], highlight: int, title: str) -> None:
    """One-series horizontal bars: the highlighted model in blue, the rest as gray context."""
    y = np.arange(len(labels))[::-1]
    colors = [SERIES[0] if i == highlight else AXIS for i in range(len(labels))]
    ax.barh(y, values, height=0.55, color=colors)
    for yi, value in zip(y, values):
        ax.text(value + max(values) * 0.02, yi, f"{value:.3f}", va="center", fontsize=8, color=INK_SECONDARY)
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", length=0, labelcolor=INK_SECONDARY)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(values) * 1.2)
    ax.set_title(title, loc="left")


def comparison_figure(name: str, entries: list[tuple[str, dict]], keys: list[tuple[str, str]], n_baselines: int) -> None:
    if len(entries) < 2:
        return
    labels = [label for label, _ in entries]
    first = [run["test"][keys[0][0]] for _, run in entries]
    highlight = n_baselines + int(np.argmax(first[n_baselines:])) if len(entries) > n_baselines else 0
    fig, axes = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 0.42 * len(entries) + 0.9), sharey=True)
    for ax, (key, title) in zip(np.atleast_1d(axes), keys):
        hbars(ax, labels, [run["test"][key] for _, run in entries], highlight, title)
    fig.tight_layout()
    save(fig, name)


def stage1_curves() -> None:
    runs = {"raw": "captions as written", "masked": "tag words masked"}
    histories = {run: load_json(RESULTS / "history" / f"stage1_bert_{run}.json") for run in runs}
    if not all(histories.values()):
        return
    fig, axes = plt.subplots(2, 2, figsize=(7, 4.4), sharex=True, sharey="row")
    for col, (run, subtitle) in enumerate(runs.items()):
        history = histories[run]
        epochs = [h["epoch"] for h in history]
        for row, (key, label) in enumerate((("macro_f1", "Macro-F1"), ("micro_f1", "Micro-F1"))):
            ax = axes[row, col]
            for i, (split, split_name) in enumerate((("val", "Validation"), ("test", "Test"))):
                ax.plot(epochs, [h[split][key] for h in history], color=SERIES[i], label=split_name)
            ax.set_ylim(bottom=0)
            if row == 0:
                ax.set_title(f"BERT, {subtitle}", loc="left")
            if col == 0:
                ax.set_ylabel(label)
            if row == 1:
                ax.set_xlabel("Epoch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.04))
    fig.tight_layout()
    save(fig, "stage1_curves")


def stage2_comparison(metrics: dict) -> None:
    candidates = [
        ("Majority class", "baseline_majority_genre"),
        ("CNN on log-mel", "baseline_cnn_genre"),
        ("GraphSAGE, segment graph", "stage2_graphsage_segment"),
        ("GAT, segment graph", "stage2_gat_segment"),
        ("GraphSAGE, chord graph", "stage2_graphsage_chord"),
        ("GAT, chord graph", "stage2_gat_chord"),
    ]
    entries = [(label, metrics[key]) for label, key in candidates if key in metrics]
    n_baselines = sum(key.startswith("baseline_") for _, key in candidates if key in metrics)
    comparison_figure("stage2_comparison", entries, [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1")], n_baselines)


def stage2_confusion(metrics: dict) -> None:
    runs = {name: run for name, run in metrics.items() if name.startswith("stage2_")}
    if not runs:
        return
    name, best = max(runs.items(), key=lambda item: item[1]["test"]["macro_f1"])
    counts = np.array(best["test_confusion"], dtype=float)
    shares = counts / counts.sum(axis=1, keepdims=True)
    classes = best["classes"]
    fig, ax = plt.subplots(figsize=(4.8, 4.1))
    image = ax.imshow(shares, cmap=BLUE, vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=40, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):  # label only the diagonal: per-genre recall
        ax.text(i, i, f"{shares[i, i]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if shares[i, i] > 0.5 else INK)
    ax.grid(False)
    ax.set_xlabel("Predicted genre")
    ax.set_ylabel("True genre")
    ax.set_title(f"{name.removeprefix('stage2_').replace('_', ', ')}: test confusion", loc="left")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Share of the true genre")
    save(fig, "stage2_confusion")


def stage3_ablation(metrics: dict) -> None:
    entries = []
    for variant, label in VARIANT_NAMES.items():
        run = pick(metrics, f"stage3_{variant}_masked_pretrained", f"stage3_{variant}_masked")
        if run:
            entries.append((label, run))
    comparison_figure("stage3_ablation", entries, [("macro_f1", "Macro-F1"), ("auc_pr", "AUC-PR")], 0)


def dominant_labels(tag_lists: list[list[str]], candidates: list[str], k: int = 3) -> tuple[list[str], list[str]]:
    """Keep the k most frequent candidate tags; each clip gets the first of them it has, or 'other'."""
    counts = {c: sum(c in tags for tags in tag_lists) for c in candidates}
    top = sorted(candidates, key=lambda c: -counts[c])[:k]
    return [next((c for c in top if c in tags), "other") for tags in tag_lists], top


def tsne_fused() -> None:
    from sklearn.manifold import TSNE

    from src.datasets import load_musiccaps

    paths = sorted(CHECKPOINTS.glob("stage3_cross_attention_masked*_test_z.npy"))
    if not paths:
        return
    z = np.load(paths[-1])
    ids = json.loads(paths[-1].with_name(paths[-1].name.replace("_test_z.npy", "_test_ids.json")).read_text())
    tags = load_musiccaps(load_config()).set_index("ytid").loc[ids, "tags"].tolist()
    xy = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(z)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.9))
    for ax, (title, candidates) in zip(axes, (("Coloured by genre tag", GENRE_TAGS), ("Coloured by mood tag", MOOD_TAGS))):
        labels, top = dominant_labels(tags, candidates)
        labels = np.array(labels)
        other = labels == "other"
        ax.scatter(xy[other, 0], xy[other, 1], s=6, color=AXIS, linewidths=0, label="other / none")
        for i, tag in enumerate(top):
            chosen = labels == tag
            ax.scatter(xy[chosen, 0], xy[chosen, 1], s=16, color=SERIES[i], edgecolors=SURFACE, linewidths=0.5, label=tag)
        for tag in top:  # name each group in ink at its center, so identity never rests on color alone
            chosen = labels == tag
            if chosen.any():
                ax.annotate(
                    tag, tuple(np.median(xy[chosen], axis=0)), ha="center", va="center", fontsize=8, color=INK,
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": SURFACE, "edgecolor": "none", "alpha": 0.85},
                )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, loc="left")
        ax.legend(loc="lower left", markerscale=1.2, handletextpad=0.2)
    fig.suptitle("t-SNE of the fused representation z (test clips)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    save(fig, "stage3_tsne")


def stage4_retrieval(metrics: dict) -> None:
    run = pick(metrics, "stage4_contrastive_pretrained", "stage4_contrastive")
    if not run:
        return
    ks = [1, 5, 10]
    x = np.arange(len(ks))
    width = 0.36
    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    for i, (direction, label) in enumerate((("text_to_audio", "Text → audio"), ("audio_to_text", "Audio → text"))):
        positions = x + (i - 0.5) * (width + 0.03)
        values = [run["test"][f"{direction}_R@{k}"] for k in ks]
        ax.bar(positions, values, width=width, color=SERIES[i], label=label)
        for position, value in zip(positions, values):
            ax.text(position, value + 0.004, f"{value:.2f}", ha="center", va="bottom", fontsize=7, color=INK_SECONDARY)
    ax.set_xticks(x, [f"R@{k}" for k in ks])
    ax.set_ylabel("Recall on the test split")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    save(fig, "stage4_retrieval")


def song_arc_diagram(dataset: str = "fma_small") -> None:
    """Draw one track's segment graph as an arc diagram: arcs join moments that sound alike."""
    cfg = load_config()
    path = resolve(cfg["paths"]["processed"]) / dataset / "segment_graphs.pt"
    if not path.exists():
        return
    graphs = torch.load(path, weights_only=False)

    def long_range_links(g) -> int:
        src, dst = g.edge_index
        similar = (g.edge_attr[:, 1] > 0) & (src < dst)
        return int(((g.segment_seconds[dst[similar], 0] - g.segment_seconds[src[similar], 0]) > 5).sum())

    key = max(graphs, key=lambda k: long_range_links(graphs[k]))
    g = graphs[key]
    times = g.segment_seconds.mean(dim=1).numpy()
    src, dst = g.edge_index.numpy()
    similar = g.edge_attr[:, 1].numpy() > 0
    cosine = g.edge_attr[:, 2].numpy()
    norm = Normalize(cfg["graph"]["similarity_threshold"], 1.0)

    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    ax.plot(times, np.zeros_like(times), color=AXIS, linewidth=1.2, zorder=1)
    for i, j, c in sorted(zip(src[similar], dst[similar], cosine[similar]), key=lambda edge: edge[2]):
        if i < j:
            x0, x1 = times[i], times[j]
            curve = Path([(x0, 0), ((x0 + x1) / 2, x1 - x0), (x1, 0)], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
            ax.add_patch(PathPatch(curve, facecolor="none", edgecolor=BLUE(norm(c)), linewidth=1.1, zorder=2))
    ax.scatter(times, np.zeros_like(times), s=14, color=SERIES[0], edgecolors=SURFACE, linewidths=0.6, zorder=3)
    ax.set_xlim(times[0] - 0.5, times[-1] + 0.5)
    ax.set_ylim(-0.6, (times[-1] - times[0]) / 2 + 0.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    ax.set_xlabel("Time in the clip (s)")
    title = f"Segment graph of FMA track {key}"
    if dataset == "fma_small":
        from src.datasets import load_fma_small

        genre = load_fma_small(cfg).set_index("track_id").at[key, "genre"]
        title += f" ({genre})"
    ax.set_title(title + ": arcs join segments that sound alike", loc="left")
    fig.colorbar(ScalarMappable(norm, BLUE), ax=ax, fraction=0.03, pad=0.01, label="Cosine similarity")
    save(fig, "song_arc_diagram")


def main() -> None:
    plt.switch_backend("Agg")  # render straight to files; notebooks keep their inline backend
    style()
    metrics = load_json(RESULTS / "metrics.json") or {}
    stage1_curves()
    stage2_comparison(metrics)
    stage2_confusion(metrics)
    stage3_ablation(metrics)
    tsne_fused()
    stage4_retrieval(metrics)
    song_arc_diagram()


if __name__ == "__main__":
    main()
