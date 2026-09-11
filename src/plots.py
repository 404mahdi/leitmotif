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
NEGATIVE = "#e34948"  # the red pole of the blue-red diverging pair, for decreases
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
    """PNG and PDF in results/plots/, plus a PDF copy in report/figures/ so the report folder compiles on its own."""
    report_figures = ROOT / "report" / "figures"
    PLOTS.mkdir(parents=True, exist_ok=True)
    report_figures.mkdir(parents=True, exist_ok=True)
    for path in (PLOTS / f"{name}.png", PLOTS / f"{name}.pdf", report_figures / f"{name}.pdf"):
        fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote results/plots/{name}.png/.pdf and report/figures/{name}.pdf")


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


STAGE2_LABELS = {
    "baseline_majority_genre": "Majority class",
    "baseline_cnn_genre": "CNN on log-mel",
    "stage2_graphsage_chord": "GraphSAGE, chord graph",
    "stage2_gat_chord": "GAT, chord graph",
    "stage2_graphsage_segment": "GraphSAGE, segment graph",
    "stage2_gat_segment": "GAT, segment graph",
    "stage2_graphsage_segment_crop50": "GraphSAGE, segment graph + crops",
    "stage2_gat_segment_crop50": "GAT, segment graph + crops",
    "stage2_graphsage_segment_cnn_nodes": "GraphSAGE + CNN node features",
    "stage2_gat_segment_cnn_nodes": "GAT + CNN node features",
}


def stage2_comparison(metrics: dict) -> None:
    entries = [(label, metrics[key]) for key, label in STAGE2_LABELS.items() if key in metrics]
    n_baselines = sum(key.startswith("baseline_") and key in metrics for key in STAGE2_LABELS)
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
    ax.set_title(f"{STAGE2_LABELS.get(name, name)}: test confusion", loc="left")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Share of the true genre")
    save(fig, "stage2_confusion")


def stage3_ablation(metrics: dict) -> None:
    entries = []
    for variant, label in VARIANT_NAMES.items():
        run = pick(metrics, f"stage3_{variant}_masked_pretrained", f"stage3_{variant}_masked")
        if run:
            entries.append((label, run))
    comparison_figure("stage3_ablation", entries, [("macro_f1", "Macro-F1"), ("auc_pr", "AUC-PR")], 0)
    stage3_tag_gains(metrics)


def stage3_tag_gains(metrics: dict, top: int = 8) -> None:
    """Per-tag change in test AP when the audio graph is fused with the caption, against BERT alone."""
    from sklearn.metrics import average_precision_score

    from src.datasets import load_musiccaps, multi_hot

    fused = [v for v in ("cross_attention", "concat") if f"stage3_{v}_masked_pretrained" in metrics]
    if "stage3_bert_only_masked_pretrained" not in metrics or not fused:
        return
    variant = max(fused, key=lambda v: metrics[f"stage3_{v}_masked_pretrained"]["test"]["macro_f1"])
    cfg = load_config()
    vocab = list(json.loads((resolve(cfg["paths"]["splits"]) / "musiccaps_tags.json").read_text(encoding="utf-8")))
    clips = load_musiccaps(cfg).set_index("ytid")
    ids = json.loads((CHECKPOINTS / f"stage3_{variant}_masked_pretrained_test_ids.json").read_text(encoding="utf-8"))
    targets = np.stack([multi_hot(clips.at[ytid, "tags"], vocab) for ytid in ids])

    def per_tag_ap(name: str) -> np.ndarray:
        probs = np.load(CHECKPOINTS / f"{name}_test_probs.npy")
        row = {ytid: i for i, ytid in enumerate(json.loads((CHECKPOINTS / f"{name}_test_ids.json").read_text(encoding="utf-8")))}
        probs = probs[[row[ytid] for ytid in ids]]
        return np.array([average_precision_score(targets[:, k], probs[:, k]) for k in range(len(vocab))])

    delta = per_tag_ap(f"stage3_{variant}_masked_pretrained") - per_tag_ap("stage3_bert_only_masked_pretrained")
    order = np.argsort(delta)
    chosen = np.concatenate([order[:top], order[-top:]])  # largest losses, then largest gains
    fig, ax = plt.subplots(figsize=(6.2, 0.28 * len(chosen) + 1.1))
    y = np.arange(len(chosen))
    # Diverging encoding: blue for tags the audio graph helps, red for tags it hurts, grey zero line
    ax.barh(y, delta[chosen], height=0.6, color=[SERIES[0] if delta[k] > 0 else NEGATIVE for k in chosen])
    ax.axvline(0, color=AXIS, linewidth=0.8)
    limit = float(np.abs(delta[chosen]).max()) * 1.4
    for yi, k in zip(y, chosen):
        value = float(delta[k])
        offset = limit * 0.02 if value >= 0 else -limit * 0.02
        ax.text(value + offset, yi, f"{value:+.2f}", va="center", ha="left" if value >= 0 else "right", fontsize=7,
                color=INK_SECONDARY)
    ax.set_xlim(-limit, limit)
    ax.set_yticks(y, [vocab[k] for k in chosen])
    ax.tick_params(axis="y", length=0, labelsize=8, labelcolor=INK_SECONDARY)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Change in test average precision")
    ax.set_title(f"Adding the audio graph: {VARIANT_NAMES[variant].lower()} minus BERT only", loc="left")
    save(fig, "stage3_tag_gains")


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
        # Legend under the axes, where it can't cover the densest clusters
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, markerscale=1.2, handletextpad=0.2,
                  columnspacing=1.0)
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
            ax.text(position, value + 0.002, f"{value:.3f}", ha="center", va="bottom", fontsize=7, color=INK_SECONDARY)
    ax.set_xticks(x, [f"R@{k}" for k in ks])
    ax.set_ylabel("Recall on the test split")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    save(fig, "stage4_retrieval")


def draw_arcs(ax, times: np.ndarray, edge_index: np.ndarray, edge_attr: np.ndarray, threshold: float,
              node_values: np.ndarray | None = None) -> Normalize:
    """Draw a segment graph as an arc diagram on `ax` and return the colour scale of the arcs.

    Nodes sit on a time axis joined by the temporal chain; each similarity edge is an arc coloured by its
    cosine similarity. When `node_values` is given, nodes are shaded by it (e.g. attention per segment).
    """
    src, dst = edge_index
    similar = edge_attr[:, 1] > 0
    norm = Normalize(threshold, 1.0)
    ax.plot(times, np.zeros_like(times), color=AXIS, linewidth=1.2, zorder=1)
    for i, j, c in sorted(zip(src[similar], dst[similar], edge_attr[similar, 2]), key=lambda edge: edge[2]):
        if i < j:
            x0, x1 = times[i], times[j]
            curve = Path([(x0, 0), ((x0 + x1) / 2, x1 - x0), (x1, 0)], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
            ax.add_patch(PathPatch(curve, facecolor="none", edgecolor=BLUE(norm(c)), linewidth=1.1, zorder=2))
    if node_values is None:
        ax.scatter(times, np.zeros_like(times), s=14, color=SERIES[0], edgecolors=SURFACE, linewidths=0.6, zorder=3)
    else:
        shade = BLUE(0.15 + 0.85 * node_values / max(float(node_values.max()), 1e-8))
        ax.scatter(times, np.zeros_like(times), s=34, color=shade, edgecolors=SURFACE, linewidths=0.8, zorder=3)
    ax.set_xlim(times[0] - 0.5, times[-1] + 0.5)
    ax.set_ylim(-0.6, (times[-1] - times[0]) / 2 + 0.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    return norm


def song_arc_diagram(dataset: str = "fma_small") -> None:
    """One track's segment graph as an arc diagram: arcs join moments that sound alike."""
    cfg = load_config()
    path = resolve(cfg["paths"]["processed"]) / dataset / "segment_graphs.pt"
    if not path.exists():
        return
    graphs = torch.load(path, weights_only=False)

    def similarity_pairs(g) -> tuple[int, int]:
        """(similarity edges, those joining moments more than 5 s apart), counting each pair once."""
        src, dst = g.edge_index
        similar = (g.edge_attr[:, 1] > 0) & (src < dst)
        gaps = g.segment_seconds[dst[similar], 0] - g.segment_seconds[src[similar], 0]
        return int(similar.sum()), int((gaps > 5).sum())

    # A readable example: clear long-range repetition, but not a clip where every moment matches every other
    counts = {k: similarity_pairs(g) for k, g in graphs.items()}
    readable = [k for k, (pairs, _) in counts.items() if 20 <= pairs <= 45]
    key = max(readable or counts, key=lambda k: counts[k][1])
    g = graphs[key]
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    norm = draw_arcs(
        ax, g.segment_seconds.mean(dim=1).numpy(), g.edge_index.numpy(), g.edge_attr.numpy(),
        cfg["graph"]["similarity_threshold"],
    )
    ax.set_xlabel("Time in the clip (s)")
    title = f"Segment graph of FMA track {key}"
    if dataset == "fma_small":
        from src.datasets import load_fma_small

        genre = load_fma_small(cfg).set_index("track_id").at[key, "genre"]
        title += f" ({genre})"
    ax.set_title(title + ": arcs join segments that sound alike", loc="left")
    fig.colorbar(ScalarMappable(norm, BLUE), ax=ax, fraction=0.03, pad=0.01, label="Cosine similarity")
    save(fig, "song_arc_diagram")


STOPWORDS = {
    "the", "and", "with", "this", "that", "these", "those", "are", "was", "were", "been", "its", "his", "her", "their",
    "they", "has", "have", "for", "from", "over", "while", "there", "which", "who", "into", "onto", "out", "can", "could",
    "would", "may", "might", "also", "very", "some", "all", "one", "two", "other", "such", "than", "then", "through",
    "throughout", "you", "your", "way", "time", "used", "probably", "consists", "contains", "song", "songs", "music",
    "track", "recording", "features", "featuring", "sound", "sounds", "like", "hear", "heard", "being", "playing",
    "played", "plays", "background", "foreground", "catch",
}


def case_studies(name: str = "stage3_cross_attention_masked_pretrained", n: int = 3) -> None:
    """Three test clips: caption words against segments under the fusion model's attention, plus the clip's graph."""
    if not (CHECKPOINTS / f"{name}.pt").exists():
        return
    from sklearn.metrics import average_precision_score

    from src.inference import Leitmotif

    lm = Leitmotif(fusion=name)
    threshold = lm.cfg["graph"]["similarity_threshold"]
    ids = json.loads((CHECKPOINTS / f"{name}_test_ids.json").read_text(encoding="utf-8"))
    probs = np.load(CHECKPOINTS / f"{name}_test_probs.npy")
    truth = np.array([[tag in lm.clips.at[ytid, "tags"] for tag in lm.vocab] for ytid in ids], dtype=int)

    # Clips the model gets right, each led by a different tag so the three cases differ
    scored = {i: average_precision_score(truth[i], probs[i]) for i in np.where(truth.sum(axis=1) >= 3)[0]}
    chosen, leads = [], set()
    for i in sorted(scored, key=scored.get, reverse=True):
        lead = lm.vocab[int(np.argmax(probs[i] * truth[i]))]
        if lead not in leads:
            chosen.append(i)
            leads.add(lead)
        if len(chosen) == n:
            break

    fig, axes = plt.subplots(n, 2, figsize=(8.2, 2.6 * n), gridspec_kw={"width_ratios": [1.3, 1]})
    records = []
    for row, i in enumerate(chosen):
        ytid = ids[i]
        clip = lm.clips.loc[ytid]
        explanation = lm.explain(clip["audio_path"], clip["caption"])
        tokens = explanation["tokens"]
        words = [k for k, tok in enumerate(tokens) if tok.isalpha() and len(tok) > 2 and tok not in STOPWORDS]
        words = sorted(sorted(words, key=lambda k: -explanation["attention"][k].max())[:10])
        attention = explanation["attention"][words]
        starts = explanation["segment_seconds"][:, 0]
        predicted = [lm.vocab[k] for k in np.argsort(-probs[i])[:4]]

        heat = axes[row, 0]
        heat.imshow(attention, aspect="auto", cmap=BLUE, vmin=0)
        heat.set_yticks(range(len(words)), [tokens[k] for k in words], fontsize=7)
        heat.set_xticks(range(0, len(starts), 4), [f"{s:g}s" for s in starts[::4]], fontsize=7)
        heat.tick_params(length=0)
        heat.grid(False)
        heat.set_title(f"Clip {ytid}. Predicted: {', '.join(predicted)}", loc="left", fontsize=8)

        arcs = axes[row, 1]
        draw_arcs(arcs, explanation["segment_seconds"].mean(axis=1), explanation["edge_index"],
                  explanation["edge_attr"], threshold, node_values=attention.mean(axis=0))
        arcs.set_title("Segment graph, nodes shaded by attention", loc="left", fontsize=8)
        records.append({
            "ytid": ytid,
            "caption": clip["caption"],
            "true_tags": [t for t in clip["tags"] if t in lm.vocab],
            "predicted_top4": predicted,
            "clip_average_precision": round(float(scored[i]), 3),
            "most_attended_segment_per_word": {
                tokens[k]: f"{starts[int(np.argmax(explanation['attention'][k]))]:.1f}s" for k in words
            },
        })
    axes[-1, 0].set_xlabel("Segment start")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.tight_layout()
    save(fig, "case_studies")
    (RESULTS / "case_studies.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


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
    case_studies()


if __name__ == "__main__":
    main()
