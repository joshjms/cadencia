"""Shared architecture for training and checkpoint inference (requires torch)."""

import torch
from torch import nn


class HitTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(20, 32)
        self.position = nn.Parameter(torch.randn(1, 100, 32) * 0.02)
        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=32, nhead=4, dim_feedforward=128, dropout=0.1,
                batch_first=True,
            ) for _ in range(2)
        ])
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = self.projection(x) + self.position
        for layer in self.encoder:
            x = layer(x)
        return self.output(x.mean(dim=1)).squeeze(-1)
