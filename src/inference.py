"""Use trained Leitmotif models on any audio file.

    from src.inference import Leitmotif

    lm = Leitmotif()
    lm.tags("song.wav", "a mellow piano ballad")         # {tag: probability}, from the stage 3 fusion model
    lm.explain("song.wav", "a mellow piano ballad")      # which segments each caption word attends to
    lm.search("lo-fi hip hop with a mellow piano loop")  # closest MusicCaps clips, from the stage 4 dual encoder
    lm.describe("song.wav")                              # the human captions closest to a new clip

Checkpoints are read from checkpoints/ as written by `python -m src.train`.
"""

import json

import numpy as np
import torch
from transformers import AutoTokenizer

from src import ROOT
from src.audio_features import extract_features, load_audio
from src.contrastive import DualEncoder
from src.datasets import load_musiccaps, mask_tags, tag_mask_pattern
from src.fusion_model import FusionTagger
from src.gnn_model import GraphEncoder
from src.graph_builder import segment_graph
from src.utils import load_config, resolve

CHECKPOINTS = ROOT / "checkpoints"


class Leitmotif:
    def __init__(
        self,
        fusion: str = "stage3_cross_attention_masked_pretrained",
        retrieval: str = "stage4_contrastive_pretrained",
        device: str | None = None,
    ):
        self.cfg = load_config()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.vocab = list(json.loads((resolve(self.cfg["paths"]["splits"]) / "musiccaps_tags.json").read_text(encoding="utf-8")))
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg["text"]["model"])
        self.mask_pattern = tag_mask_pattern(self.vocab)
        self.fusion_name, self.retrieval_name = fusion, retrieval
        self._fusion = None
        self._retrieval = None
        self._clips = None

    # ------------------------------------------------------------ loading

    def _checkpoint(self, name: str) -> tuple[dict, dict, dict]:
        state = torch.load(CHECKPOINTS / f"{name}.pt", map_location="cpu")
        run = json.loads((CHECKPOINTS / f"{name}_args.json").read_text(encoding="utf-8"))
        stats = torch.load(CHECKPOINTS / f"{name}_node_stats.pt", map_location="cpu")
        return state, run, stats

    @staticmethod
    def _graph_encoder(state: dict, run: dict) -> GraphEncoder:
        gcfg = run["config"]["train"]["gnn_genre"]
        in_dim = state["graph.project.weight"].shape[1]
        return GraphEncoder(in_dim, gcfg["hidden"], gcfg["layers"], run["args"]["arch"], gcfg["dropout"], edge_dim=3)

    @property
    def fusion(self):
        if self._fusion is None:
            state, run, stats = self._checkpoint(self.fusion_name)
            args, tcfg = run["args"], run["config"]["train"]["fusion"]
            encoder = None if args["variant"] == "bert_only" else self._graph_encoder(state, run)
            model = FusionTagger(
                args["variant"], run["config"]["text"]["model"], encoder, len(self.vocab),
                tcfg["dim"], tcfg["heads"], tcfg["dropout"], tcfg["trainable_bert_layers"],
            )
            model.load_state_dict(state)
            self._fusion = (model.to(self.device).eval(), stats, args)
        return self._fusion

    @property
    def retrieval(self):
        if self._retrieval is None:
            state, run, stats = self._checkpoint(self.retrieval_name)
            tcfg = run["config"]["train"]["contrastive"]
            model = DualEncoder(
                run["config"]["text"]["model"], self._graph_encoder(state, run),
                tcfg["dim"], tcfg["temperature"], tcfg["trainable_bert_layers"],
            )
            model.load_state_dict(state)
            index = torch.load(CHECKPOINTS / f"{self.retrieval_name}_index.pt", map_location="cpu")
            self._retrieval = (model.to(self.device).eval(), stats, index)
        return self._retrieval

    @property
    def clips(self):
        """MusicCaps metadata indexed by YouTube id."""
        if self._clips is None:
            self._clips = load_musiccaps(self.cfg).set_index("ytid")
        return self._clips

    # ------------------------------------------------------------ inputs

    def graph(self, audio_path, stats: dict, offset: float = 0.0, seconds: float | None = 10.0):
        """Segment graph of `seconds` of audio starting at `offset`, normalized like the training graphs."""
        y = load_audio(audio_path, self.cfg["audio"]["sample_rate"], max_seconds=None if seconds is None else offset + seconds)
        y = y[int(offset * self.cfg["audio"]["sample_rate"]):]
        data = segment_graph(extract_features(y, self.cfg["audio"]), self.cfg["segments"], self.cfg["graph"])
        data.x = (data.x - stats["mean"]) / stats["std"]
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
        return data

    def _with_caption(self, data, caption: str, masked: bool):
        text = mask_tags(caption, self.mask_pattern) if masked else caption
        encoded = self.tokenizer(
            [text], padding="max_length", truncation=True, max_length=self.cfg["text"]["max_length"], return_tensors="pt"
        )
        data.input_ids, data.attention_mask = encoded["input_ids"], encoded["attention_mask"]
        return data.to(self.device)

    def _autocast(self):
        return torch.autocast(self.device.type, dtype=torch.bfloat16)

    # ------------------------------------------------------------ stage 3

    @torch.no_grad()
    def tags(self, audio_path, caption: str = "", top: int = 10, **window) -> dict[str, float]:
        """The `top` most likely tags for a clip, given its audio and an optional caption."""
        model, stats, args = self.fusion
        data = self._with_caption(self.graph(audio_path, stats, **window), caption, args["text"] == "masked")
        with self._autocast():
            probs = torch.sigmoid(model(data).float())[0].cpu().numpy()
        return {self.vocab[i]: float(probs[i]) for i in np.argsort(-probs)[:top]}

    @torch.no_grad()
    def explain(self, audio_path, caption: str, **window) -> dict:
        """Text-to-segment attention of the cross-attention model: a (tokens, segments) matrix."""
        model, stats, args = self.fusion
        if args["variant"] != "cross_attention":
            raise ValueError("explain() needs the cross-attention fusion model")
        data = self._with_caption(self.graph(audio_path, stats, **window), caption, args["text"] == "masked")
        with self._autocast():
            _, attention = model.fuse(data, return_attention=True)
        keep = data.attention_mask[0].bool()
        return {
            "tokens": self.tokenizer.convert_ids_to_tokens(data.input_ids[0][keep].tolist()),
            "segment_seconds": data.segment_seconds.cpu().numpy(),
            "attention": attention["text_to_graph"][0, keep.cpu()].float().cpu().numpy(),
        }

    # ------------------------------------------------------------ stage 4

    def _result(self, ytid: str, score: float) -> dict:
        clip = self.clips.loc[ytid]
        return {
            "ytid": ytid,
            "score": round(score, 4),
            "caption": clip["caption"],
            "url": f"https://www.youtube.com/watch?v={ytid}&t={int(clip['start_s'])}s",
        }

    @torch.no_grad()
    def search(self, query: str, k: int = 5) -> list[dict]:
        """MusicCaps clips whose audio graphs are closest to a text query."""
        model, _, index = self.retrieval
        encoded = self.tokenizer([query], truncation=True, max_length=self.cfg["text"]["max_length"], return_tensors="pt").to(self.device)
        with self._autocast():
            q = model.embed_text(encoded["input_ids"], encoded["attention_mask"]).float().cpu()
        scores = (index["graph_embeddings"] @ q.T).squeeze(1)
        top = torch.topk(scores, k)
        return [self._result(index["ytids"][i], float(s)) for s, i in zip(top.values, top.indices)]

    @torch.no_grad()
    def describe(self, audio_path, k: int = 3, **window) -> list[dict]:
        """The MusicCaps captions closest to a new clip's audio graph."""
        model, stats, index = self.retrieval
        data = self.graph(audio_path, stats, **window).to(self.device)
        with self._autocast():
            g = model.embed_graph(data).float().cpu()
        scores = (index["text_embeddings"] @ g.T).squeeze(1)
        top = torch.topk(scores, k)
        return [self._result(index["ytids"][i], float(s)) for s, i in zip(top.values, top.indices)]
