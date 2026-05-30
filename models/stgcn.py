import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import A3TGCN


class TrafficForecaster(nn.Module):
    """Attention Temporal Graph Convolutional Network for traffic speed forecasting."""

    def __init__(self, node_features: int, out_steps: int, periods: int = 12):
        super().__init__()
        self.tgcn = A3TGCN(in_channels=node_features, out_channels=32, periods=periods)
        self.linear = nn.Linear(32, out_steps)

    def forward(self, x, edge_index, edge_weight=None):
        # x: [nodes, features, timesteps]
        h = self.tgcn(x, edge_index, edge_weight)   # [nodes, 32]
        return self.linear(h)                        # [nodes, out_steps]
