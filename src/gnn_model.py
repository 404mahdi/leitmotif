"""GraphSAGE and GAT encoders with mean-pool readout (stage 2)."""

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, SAGEConv, global_mean_pool


class GraphEncoder(nn.Module):
    """Project node features, run L message-passing layers, and mean-pool: g = (1/|V|) sum_i h_i^(L).

    graphsage: h_i' = W1 h_i + W2 mean_{j in N(i)} h_j, which equals W [h_i || mean_j h_j]
    gat:       GATv2 attention over neighbors, conditioned on the edge attributes
    Every layer adds a residual connection and LayerNorm so deeper stacks train stably.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int = 256,
        layers: int = 3,
        arch: str = "graphsage",
        dropout: float = 0.3,
        heads: int = 4,
        edge_dim: int | None = None,
    ):
        super().__init__()
        if arch not in ("graphsage", "gat"):
            raise ValueError(f"unknown GNN architecture: {arch}")
        self.arch, self.dim = arch, hidden
        self.project = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList(
            GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=edge_dim) if arch == "gat" else SAGEConv(hidden, hidden)
            for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)

    def node_states(self, data) -> torch.Tensor:
        h = self.project(data.x)
        for conv, norm in zip(self.convs, self.norms):
            out = conv(h, data.edge_index, data.edge_attr) if self.arch == "gat" else conv(h, data.edge_index)
            h = norm(h + self.dropout(torch.relu(out)))
        return h

    def forward(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the graph embedding g (n_graphs, hidden) and the node states h (n_nodes, hidden)."""
        h = self.node_states(data)
        return global_mean_pool(h, data.batch), h


class GraphClassifier(nn.Module):
    def __init__(self, encoder: GraphEncoder, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(encoder.dim, n_classes))

    def forward(self, data) -> torch.Tensor:
        g, _ = self.encoder(data)
        return self.head(g)
