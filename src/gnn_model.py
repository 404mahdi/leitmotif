"""GraphSAGE and GAT encoders with mean-pool readout (stage 2)."""

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, SAGEConv, global_mean_pool


class SegmentCNN(nn.Module):
    """Learned node features: a small CNN embeds each segment's log-mel patch (bands x frames)."""

    def __init__(self, out_dim: int = 128, channels: tuple[int, ...] = (32, 64, 128)):
        super().__init__()
        blocks, c_in = [], 1
        for c_out in channels:
            blocks += [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*blocks)
        self.out = nn.Linear(2 * c_in, out_dim)
        self.out_dim = out_dim

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """(n_segments, n_mels, frames) -> (n_segments, out_dim) via global mean and max pooling."""
        x = self.features(patches.unsqueeze(1).float())  # patches may be stored as float16 to save memory
        return self.out(torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1))


class GraphEncoder(nn.Module):
    """Project node features, run L message-passing layers, and mean-pool: g = (1/|V|) sum_i h_i^(L).

    graphsage: h_i' = W1 h_i + W2 mean_{j in N(i)} h_j, which equals W [h_i || mean_j h_j]
    gat:       GATv2 attention over neighbors, conditioned on the edge attributes
    Every layer adds a residual connection and LayerNorm so deeper stacks train stably. With `segment_cnn`,
    each node's handcrafted features are concatenated with a CNN embedding of its log-mel patch.
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
        segment_cnn: SegmentCNN | None = None,
    ):
        super().__init__()
        if arch not in ("graphsage", "gat"):
            raise ValueError(f"unknown GNN architecture: {arch}")
        self.arch, self.dim = arch, hidden
        self.segment_cnn = segment_cnn
        node_dim = in_dim + (segment_cnn.out_dim if segment_cnn is not None else 0)
        self.project = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList(
            GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=edge_dim) if arch == "gat" else SAGEConv(hidden, hidden)
            for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)

    def node_states(self, data) -> torch.Tensor:
        x = data.x
        if self.segment_cnn is not None:
            x = torch.cat([x, self.segment_cnn(data.patches)], dim=1)
        h = self.project(x)
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
