"""GNN + BERT fusion (stage 3): BERT-only, GNN-only, early concat and cross-attention variants."""

import torch
from torch import nn
from torch_geometric.utils import to_dense_batch

from src.bert_encoder import TextEncoder
from src.gnn_model import GraphEncoder

VARIANTS = ("bert_only", "gnn_only", "concat", "cross_attention")


class FusionTagger(nn.Module):
    """Predict tags from a clip's structure graph together with its caption.

    bert_only        z = t                         (projected caption [CLS] vector)
    gnn_only         z = g                         (projected mean-pooled graph embedding)
    concat           z = [g ; t]
    cross_attention  z = [g ; Attn(g -> H_text) ; t ; Attn(H_text -> H_graph)]

    The first attention term is the handout's A = softmax(QK^T / sqrt(d)) with Q = g W_Q and
    K = H_text W_K. The second runs the other way: every caption token attends over the song's
    segments, so its attention map shows which moments of the clip each word lines up with.
    """

    def __init__(
        self,
        variant: str,
        text_model: str,
        graph_encoder: GraphEncoder | None,
        n_tags: int,
        dim: int = 256,
        heads: int = 4,
        dropout: float = 0.2,
        trainable_bert_layers: int = 4,
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown fusion variant: {variant}")
        self.variant = variant
        self.uses_text = variant != "gnn_only"
        self.uses_graph = variant != "bert_only"
        if self.uses_text:
            self.text = TextEncoder(text_model)
            self.text.freeze_below(trainable_bert_layers)
            self.text_proj = nn.Linear(self.text.dim, dim)
        if self.uses_graph:
            self.graph = graph_encoder
            self.graph_proj = nn.Linear(graph_encoder.dim, dim)
        if variant == "cross_attention":
            self.graph_to_text = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
            self.text_to_graph = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        parts = {"bert_only": 1, "gnn_only": 1, "concat": 2, "cross_attention": 4}[variant]
        self.head = nn.Sequential(nn.Linear(parts * dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, n_tags))

    def fuse(self, batch, return_attention: bool = False):
        """The fused representation z, optionally with the attention maps used to build it."""
        parts, attention = [], {}
        if self.uses_graph:
            g, h = self.graph(batch)
            g = self.graph_proj(g)
            parts.append(g)
        if self.uses_text:
            tokens, _ = self.text(batch.input_ids, batch.attention_mask)
            tokens = self.text_proj(tokens)
            t = tokens[:, 0]
            parts.append(t)
        if self.variant == "cross_attention":
            text_padding = batch.attention_mask == 0
            nodes, node_mask = to_dense_batch(self.graph_proj(h), batch.batch)
            text_context, graph_to_text = self.graph_to_text(
                g.unsqueeze(1), tokens, tokens, key_padding_mask=text_padding, need_weights=return_attention
            )
            word_context, text_to_graph = self.text_to_graph(
                tokens, nodes, nodes, key_padding_mask=~node_mask, need_weights=return_attention
            )
            keep = batch.attention_mask.unsqueeze(-1).to(word_context.dtype)
            graph_context = (word_context * keep).sum(dim=1) / keep.sum(dim=1)
            parts = [g, text_context.squeeze(1), t, graph_context]
            if return_attention:
                attention = {
                    "graph_to_text": graph_to_text.squeeze(1),  # (batch, tokens)
                    "text_to_graph": text_to_graph,  # (batch, tokens, segments)
                    "node_mask": node_mask,
                }
        z = torch.cat(parts, dim=1)
        return (z, attention) if return_attention else z

    def forward(self, batch) -> torch.Tensor:
        return self.head(self.fuse(batch))
