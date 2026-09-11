"""Dual-encoder GNN + BERT trained with InfoNCE for text-to-music retrieval (stage 4)."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from src.bert_encoder import TextEncoder
from src.gnn_model import GraphEncoder


class DualEncoder(nn.Module):
    """Maps a clip's structure graph and a caption into one shared, L2-normalized space."""

    def __init__(
        self,
        text_model: str,
        graph_encoder: GraphEncoder,
        dim: int = 256,
        temperature: float = 0.07,
        trainable_bert_layers: int = 4,
    ):
        super().__init__()
        self.graph = graph_encoder
        self.text = TextEncoder(text_model)
        self.text.freeze_below(trainable_bert_layers)
        self.graph_proj = nn.Sequential(nn.Linear(graph_encoder.dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.text_proj = nn.Sequential(nn.Linear(self.text.dim, dim), nn.GELU(), nn.Linear(dim, dim))
        # 1 / tau, learned in log space as in CLIP and starting from the configured temperature
        self.log_scale = nn.Parameter(torch.tensor(math.log(1 / temperature)))

    def embed_graph(self, batch) -> torch.Tensor:
        g, _ = self.graph(batch)
        return F.normalize(self.graph_proj(g), dim=-1)

    def embed_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        _, cls = self.text(input_ids, attention_mask)
        return F.normalize(self.text_proj(cls), dim=-1)

    @property
    def temperature(self) -> float:
        return float(1 / self.log_scale.detach().exp().clamp(max=100))

    def forward(self, batch) -> torch.Tensor:
        """S_ij = sim(g_i, t_j) / tau for every clip i and caption j in the batch."""
        g = self.embed_graph(batch)
        t = self.embed_text(batch.input_ids, batch.attention_mask)
        return g @ t.T * self.log_scale.exp().clamp(max=100)


def info_nce(similarity: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE: each clip must pick out its own caption from the batch, and each caption its clip."""
    targets = torch.arange(len(similarity), device=similarity.device)
    return (F.cross_entropy(similarity, targets) + F.cross_entropy(similarity.T, targets)) / 2
