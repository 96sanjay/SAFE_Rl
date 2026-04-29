"""Beta distribution SAC actor for exact continuous action masking.

Replaces GaussianSACActor (Gaussian + tanh squashing) with Beta(alpha, beta)
distribution that has exact support on (0, 1). State-dependent bounds [l(s), u(s)]
are applied in the environment wrapper, not the actor.

References:
  - Stolz et al. (2024) "Excluding the Irrelevant" NeurIPS 2024
  - Figurnov et al. (2018) "Implicit Reparameterization Gradients" NeurIPS 2018
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BetaSACActor(nn.Module):
    """SAC actor with Beta distribution — exact bounded support on (0, 1).

    The network outputs (alpha, beta) parameters per action dimension.
    Sampling uses PyTorch's implicit reparameterization (Figurnov et al. 2018).
    softplus(x) + 0.5 ensures alpha, beta >= 0.5, allowing bimodal (arcsine)
    distributions for bang-bang control.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: tuple[int, ...] = (256, 256),
        activation: str = 'relu',
    ) -> None:
        super().__init__()
        act_fn = nn.ReLU if activation == 'relu' else nn.Tanh
        layers: list[nn.Module] = []
        d = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(d, h))
            layers.append(act_fn())
            d = h
        layers.append(nn.Linear(d, act_dim * 2))
        self.net = nn.Sequential(*layers)
        self.act_dim = act_dim

        # Cached distribution params (set by predict, used by log_prob)
        self._alpha: torch.Tensor | None = None
        self._beta: torch.Tensor | None = None

    def _get_dist_params(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(obs)
        alpha_raw, beta_raw = out.chunk(2, dim=-1)
        alpha = F.softplus(alpha_raw) + 0.5
        beta = F.softplus(beta_raw) + 0.5
        return alpha, beta

    def predict(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        """Sample action x in (0, 1) from Beta(alpha, beta)."""
        self._alpha, self._beta = self._get_dist_params(obs)
        dist = torch.distributions.Beta(self._alpha, self._beta)
        if deterministic:
            x = self._alpha / (self._alpha + self._beta)  # Beta mean
        else:
            x = dist.rsample()
        x = x.clamp(1e-6, 1 - 1e-6)
        return x

    def log_prob(
        self, x: torch.Tensor, obs: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute log_prob in x-space (no Jacobian correction).

        Uses cached alpha/beta from predict() if obs=None.
        If obs is provided, recomputes alpha/beta (safe for arbitrary call order).
        """
        if obs is not None:
            alpha, beta = self._get_dist_params(obs)
        else:
            alpha, beta = self._alpha, self._beta
        x = x.clamp(1e-6, 1 - 1e-6)
        dist = torch.distributions.Beta(alpha, beta)
        return dist.log_prob(x).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        """Closed-form Beta entropy (per spec: no Jacobian)."""
        dist = torch.distributions.Beta(self._alpha, self._beta)
        return dist.entropy().sum(dim=-1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass for compatibility. Returns deterministic action."""
        return self.predict(obs, deterministic=True)
