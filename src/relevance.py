"""Automatic relevance scores for retrieved clips, standing in for a human listening test.

    uv run python -m src.relevance

For every caption query in results/retrieval_examples/examples.json, each of the top-3 retrieved clips and three
random test clips is scored 1-5 by three judges that share no weights with the Leitmotif models:

- tags:    1 + 4 x Jaccard overlap between the query clip's and the candidate clip's tags (50-tag vocabulary)
- caption: similarity of the query caption and the candidate's caption under a sentence encoder (all-MiniLM-L6-v2)
- audio:   similarity of the query caption and the candidate's audio under a CLAP audio-text model

Caption and audio similarities become 1 + 4 x p, where p is the share of random test pairs that score lower: 3 means
"as similar as a typical random pair" and 5 means "more similar than every random pair". The random candidates give
the same judges' scores for clips chosen without the model.
"""

import json

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, ClapModel, ClapProcessor

from src.datasets import load_musiccaps
from src.evaluate import RESULTS
from src.utils import load_config, resolve

SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# laion/larger_clap_music would suit music better, but under transformers 5 its text tower returns the
# same embedding for every caption, so the general-audio CLAP checkpoint is used instead.
CLAP_MODEL = "laion/clap-htsat-unfused"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REFERENCE_PAIRS = 1000
JUDGES = ("tags", "caption", "audio")


def pooled(output) -> torch.Tensor:
    """transformers 5 returns a model output from get_*_features; older versions return the tensor itself."""
    return output if torch.is_tensor(output) else output.pooler_output


@torch.no_grad()
def sentence_embeddings(texts: list[str]) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(SENTENCE_MODEL)
    model = AutoModel.from_pretrained(SENTENCE_MODEL).to(DEVICE).eval()
    out = []
    for i in range(0, len(texts), 64):
        encoded = tokenizer(texts[i : i + 64], padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
        hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        out.append(F.normalize((hidden * mask).sum(dim=1) / mask.sum(dim=1), dim=-1).cpu())
    return torch.cat(out)


class Clap:
    def __init__(self):
        self.model = ClapModel.from_pretrained(CLAP_MODEL).to(DEVICE).eval()
        self.processor = ClapProcessor.from_pretrained(CLAP_MODEL)

    @torch.no_grad()
    def text(self, texts: list[str]) -> torch.Tensor:
        out = []
        for i in range(0, len(texts), 32):
            inputs = self.processor(text=texts[i : i + 32], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            out.append(F.normalize(pooled(self.model.get_text_features(**inputs)).float(), dim=-1).cpu())
        embeddings = torch.cat(out)
        if len(embeddings) > 1 and embeddings.std(dim=0).mean() < 1e-3:
            raise RuntimeError(f"{CLAP_MODEL} gives the same text embedding for every caption; it did not load correctly")
        return embeddings

    @torch.no_grad()
    def audio(self, paths: list[str]) -> torch.Tensor:
        out = []
        for i in range(0, len(paths), 16):
            waves = [librosa.load(path, sr=48000, mono=True)[0] for path in paths[i : i + 16]]
            inputs = self.processor(audio=waves, sampling_rate=48000, return_tensors="pt").to(DEVICE)
            out.append(F.normalize(pooled(self.model.get_audio_features(**inputs)).float(), dim=-1).cpu())
        return torch.cat(out)


def to_scale(value: float, reference: np.ndarray) -> float:
    """1 + 4 x (share of reference scores below value)."""
    return round(1 + 4 * float((reference < value).mean()), 2)


def main() -> None:
    cfg = load_config()
    examples = json.loads((RESULTS / "retrieval_examples" / "examples.json").read_text(encoding="utf-8"))
    vocab = set(json.loads((resolve(cfg["paths"]["splits"]) / "musiccaps_tags.json").read_text(encoding="utf-8")))
    clips = load_musiccaps(cfg).set_index("ytid")
    test = clips[(clips["split"] == "test") & clips["has_audio"]].index
    rng = np.random.default_rng(cfg["seed"])

    candidates = []
    for example in examples:
        query = example["query_ytid"]
        candidates += [(query, hit["ytid"], f"rank {rank}") for rank, hit in enumerate(example["top"], 1)]
        candidates += [(query, str(ytid), "random") for ytid in rng.choice(test[test != query], size=3, replace=False)]
    queries = sorted({query for query, _, _ in candidates})

    # Random test pairs define what "typical similarity" means for the caption and audio judges
    a, b = rng.choice(len(test), REFERENCE_PAIRS), rng.choice(len(test), REFERENCE_PAIRS)
    reference = [(test[i], test[j]) for i, j in zip(a, b) if i != j]

    caption_ids = sorted({y for pair in candidates for y in pair[:2]} | {y for pair in reference for y in pair})
    sentence = dict(zip(caption_ids, sentence_embeddings([clips.at[y, "caption"] for y in caption_ids])))
    caption_reference = np.array([float(sentence[x] @ sentence[y]) for x, y in reference])

    clap = Clap()
    text_ids = sorted(set(queries) | {x for x, _ in reference})
    audio_ids = sorted({c for _, c, _ in candidates} | set(queries) | {y for _, y in reference})
    clap_text = dict(zip(text_ids, clap.text([clips.at[y, "caption"] for y in text_ids])))
    clap_audio = dict(zip(audio_ids, clap.audio([str(clips.at[y, "audio_path"]) for y in audio_ids])))
    audio_reference = np.array([float(clap_text[x] @ clap_audio[y]) for x, y in reference])

    def tags_of(ytid: str) -> set[str]:
        return {tag for tag in clips.at[ytid, "tags"] if tag in vocab}

    scores = []
    for query, candidate, group in candidates:
        union = tags_of(query) | tags_of(candidate)
        jaccard = len(tags_of(query) & tags_of(candidate)) / len(union) if union else 0.0
        scores.append({
            "query_ytid": query,
            "candidate_ytid": candidate,
            "group": group,
            "tags": round(1 + 4 * jaccard, 2),
            "caption": to_scale(float(sentence[query] @ sentence[candidate]), caption_reference),
            "audio": to_scale(float(clap_text[query] @ clap_audio[candidate]), audio_reference),
        })

    summary = {}
    for group in ("rank 1", "rank 2", "rank 3", "random"):
        members = [s for s in scores if s["group"] == group]
        summary[group] = {judge: round(float(np.mean([s[judge] for s in members])), 2) for judge in JUDGES}
    retrieved = [s for s in scores if s["group"] != "random"]
    summary["retrieved, all ranks"] = {judge: round(float(np.mean([s[judge] for s in retrieved])), 2) for judge in JUDGES}
    summary["true clip, audio judge"] = round(float(np.mean(
        [to_scale(float(clap_text[q] @ clap_audio[q]), audio_reference) for q in queries]
    )), 2)

    report = {
        "judges": {
            "tags": "1 + 4 x Jaccard overlap of 50-vocabulary tags",
            "caption": f"{SENTENCE_MODEL} caption-caption cosine, percentile among {len(reference)} random test pairs",
            "audio": f"{CLAP_MODEL} caption-audio cosine, percentile among {len(reference)} random test pairs",
        },
        "summary": summary,
        "scores": scores,
    }
    path = RESULTS / "retrieval_examples" / "relevance.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{'':24s}" + "".join(f"{judge:>10s}" for judge in JUDGES))
    for group, values in summary.items():
        if isinstance(values, dict):
            print(f"{group:24s}" + "".join(f"{values[judge]:10.2f}" for judge in JUDGES))
    print(f"true clip, audio judge: {summary['true clip, audio judge']}")
    print(f"wrote {path.relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    main()
