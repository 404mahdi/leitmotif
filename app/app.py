"""Leitmotif web demo: search music by description, and look inside a clip.

    uv sync --group demo
    uv run --group demo python app/app.py

Needs the trained stage 3 and stage 4 checkpoints in checkpoints/ (see the README).
"""

import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from src.inference import Leitmotif  # noqa: E402
from src.plots import BLUE, STOPWORDS, draw_arcs, style  # noqa: E402

plt.switch_backend("Agg")
style()
lm = Leitmotif()


def clip_cards(results: list[dict]) -> str:
    """Retrieved MusicCaps clips as embedded YouTube players that start at the labelled 10 seconds."""
    cards = []
    for result in results:
        start = int(lm.clips.at[result["ytid"], "start_s"])
        cards.append(
            f'<div style="display:flex;gap:16px;align-items:flex-start;margin-bottom:18px;flex-wrap:wrap">'
            f'<iframe width="320" height="180" src="https://www.youtube-nocookie.com/embed/{result["ytid"]}'
            f'?start={start}&end={start + 10}" title="MusicCaps clip {result["ytid"]}" allowfullscreen></iframe>'
            f'<div style="flex:1;min-width:240px"><p style="margin:0 0 6px"><b>Similarity {result["score"]:.3f}</b> · '
            f'<a href="{result["url"]}" target="_blank" rel="noopener">open on YouTube</a></p>'
            f'<p style="margin:0">{html.escape(result["caption"])}</p></div></div>'
        )
    return "\n".join(cards)


def search(query: str, k: float) -> str:
    if not query.strip():
        raise gr.Error("Describe the music you want to find.")
    return clip_cards(lm.search(query, k=int(k)))


def structure_figure(audio_path: str, caption: str, window: dict):
    """Caption words against segments (when there is a caption) next to the clip's arc diagram."""
    explanation = lm.explain(audio_path, caption, **window)
    tokens = explanation["tokens"]
    words = [i for i, tok in enumerate(tokens) if tok.isalpha() and len(tok) > 2 and tok not in STOPWORDS]
    words = sorted(sorted(words, key=lambda i: -explanation["attention"][i].max())[:12])
    times = explanation["segment_seconds"].mean(axis=1)
    threshold = lm.cfg["graph"]["similarity_threshold"]

    if not words:
        fig, arcs = plt.subplots(figsize=(8, 2.8))
        draw_arcs(arcs, times, explanation["edge_index"], explanation["edge_attr"], threshold)
        arcs.set_title("Segment graph: arcs join moments that sound alike", loc="left")
        arcs.set_xlabel("Time (s)")
        return fig

    attention = explanation["attention"][words]
    fig, (heat, arcs) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.2, 1]})
    heat.imshow(attention, aspect="auto", cmap=BLUE, vmin=0)
    heat.set_yticks(range(len(words)), [tokens[i] for i in words], fontsize=8)
    starts = explanation["segment_seconds"][:, 0]
    heat.set_xticks(range(0, len(starts), 4), [f"{s:g}s" for s in starts[::4]], fontsize=8)
    heat.tick_params(length=0)
    heat.grid(False)
    heat.set_title("Where each word points in the clip", loc="left")
    draw_arcs(arcs, times, explanation["edge_index"], explanation["edge_attr"], threshold,
              node_values=attention.mean(axis=0))
    arcs.set_title("Segment graph, shaded by attention", loc="left")
    arcs.set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def analyze(audio_path: str | None, caption: str, offset: float):
    if not audio_path:
        raise gr.Error("Upload an audio file or pick one of the examples.")
    window = {"offset": float(offset), "seconds": 10.0}
    tags = lm.tags(audio_path, caption or "", top=10, **window)
    figure = structure_figure(audio_path, caption or "", window)
    nearest = clip_cards(lm.describe(audio_path, k=3, **window))
    return tags, figure, nearest


def example_clips(n: int = 3) -> list[list]:
    """Test clips used in the case studies when available, otherwise the first test clips with audio."""
    chosen = []
    case_studies = ROOT / "results" / "case_studies.json"
    if case_studies.exists():
        chosen = [record["ytid"] for record in json.loads(case_studies.read_text(encoding="utf-8"))]
    if len(chosen) < n:
        test = lm.clips[(lm.clips["split"] == "test") & lm.clips["has_audio"]]
        chosen += [ytid for ytid in test.index[:n] if ytid not in chosen]
    return [[str(lm.clips.at[ytid, "audio_path"]), lm.clips.at[ytid, "caption"], 0] for ytid in chosen[:n]]


with gr.Blocks(title="Leitmotif") as demo:
    gr.Markdown(
        "# Leitmotif\n"
        "Music understood two ways: as a **graph of its own structure** (one-second segments joined to the moments "
        "that follow and the moments that sound alike) and through the **words people use to describe it**. "
        "A graph neural network reads the structure, BERT reads the text. "
        "[Code and report](https://github.com/404mahdi/leitmotif)"
    )
    with gr.Tab("Search by description"):
        query = gr.Textbox(label="Describe the music", placeholder="a calm acoustic guitar melody with soft vocals")
        count = gr.Slider(1, 10, value=5, step=1, label="How many clips")
        find = gr.Button("Search", variant="primary")
        found = gr.HTML()
        gr.Examples(
            [["a calm acoustic guitar melody"], ["aggressive distorted electric guitars with fast drums"],
             ["an upbeat electronic dance track with a heavy bass"], ["a sad piano ballad with strings"]],
            inputs=[query],
        )
        find.click(search, [query, count], found)
        query.submit(search, [query, count], found)

    with gr.Tab("Analyze a clip"):
        with gr.Row():
            with gr.Column():
                audio = gr.Audio(sources=["upload"], type="filepath", label="Audio (10 seconds are used)")
                caption = gr.Textbox(label="Caption (optional)", lines=3,
                                     placeholder="Describe the clip to see which moments each word points to")
                offset = gr.Slider(0, 60, value=0, step=1, label="Start at (seconds)")
                run = gr.Button("Analyze", variant="primary")
            with gr.Column():
                tags = gr.Label(num_top_classes=10, label="Predicted tags")
        structure = gr.Plot(label="Structure and attention")
        described = gr.HTML(label="Closest human captions")
        gr.Examples(example_clips(), inputs=[audio, caption, offset])
        run.click(analyze, [audio, caption, offset], [tags, structure, described])


if __name__ == "__main__":
    demo.launch()
