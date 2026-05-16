"""Student world model.

Residual MLP + GRUCell; tuned for long open-loop rollouts on InvertedPendulum.
"""

from __future__ import annotations

import torch
from torch import nn


class _ResidualFF(nn.Module):
    """Pre-norm feedforward block with residual."""

    def __init__(self, dim: int, expansion: int = 2):
        super().__init__()
        hidden = int(dim * expansion)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        return x + y


class StudentWorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int = 4,
        act_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 2,
        use_gru: bool = False,
        delta_limit: float = 3.0,
        ff_expansion: int = 2,
    ):
        super().__init__()
        self.use_gru = bool(use_gru)
        self.delta_limit = float(delta_limit)
        in_dim = obs_dim + act_dim
        self.in_proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        depth = max(1, int(num_layers))
        self.blocks = nn.ModuleList([_ResidualFF(hidden_dim, expansion=int(ff_expansion)) for _ in range(depth)])
        self.gru = nn.GRUCell(hidden_dim, hidden_dim) if self.use_gru else None
        self.pre_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.head = nn.Linear(hidden_dim, obs_dim)
        nn.init.zeros_(self.head.bias)
        nn.init.uniform_(self.head.weight, -0.01, 0.01)

    def initial_hidden(self, batch_size: int, device: torch.device):
        if not self.use_gru:
            return None
        return torch.zeros(batch_size, self.gru.hidden_size, device=device)

    def forward(self, obs_norm: torch.Tensor, act_norm: torch.Tensor, hidden=None):
        x = self.in_proj(torch.cat([obs_norm, act_norm], dim=-1))
        for block in self.blocks:
            x = block(x)
        if self.gru is not None:
            if hidden is None:
                hidden = self.initial_hidden(obs_norm.shape[0], obs_norm.device)
            hidden = self.gru(x, hidden)
            x = hidden
        x = self.pre_head(x)
        raw_delta = self.head(x)
        delta = self.delta_limit * torch.tanh(raw_delta / self.delta_limit)
        return delta, hidden
