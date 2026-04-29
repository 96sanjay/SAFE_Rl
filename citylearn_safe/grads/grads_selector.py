"""GradS: Gradient Shaping for Multi-Constraint Safe RL.

Pure algorithm — no OmniSafe dependency.
Implements candidate set construction from Yao et al. 2024 (L4DC).

Paper-faithful: lambda pre-scaling, uniform sampling, no 1/(1+λ).
"""
from __future__ import annotations

import numpy as np
import torch


class GradSSelector:
    """Select which constraint gradient to apply via GradS (Yao et al. 2024).

    Paper-faithful algorithm:
        1. Pre-scale gradients by lambda: gᵢ = λᵢ·∇Vcᵢ  (Alg 1, line 2)
        2. Compute cosine similarity on lambda-scaled gradients
        3. Random permutation (fair anchor access)
        4. Build candidate set: -σ < sim(i,j) < κ  (Alg 1, line 6)
        5. Sample uniformly: gc ~ uniform(G)  (Alg 1, line 8)
        6. Return: ∇Jc = gc·|G|/N  (Alg 1, line 9)

    Sampling strategies:
        - 'uniform': equal probability over candidate set (paper's approach)
        - 'lambda': softmax(lambda_i) — biased toward most-violated (official code)
        - 'lambda_normalized': softmax(lambda_i / cost_limit_i)

    Args:
        num_costs: Number of constraints (default 5).
        sim_threshold: Kappa — above this, gradients are redundant (default 0.8).
        conflict_threshold: Sigma — below -sigma, gradients conflict (default 0.999).
        sampling: Sampling strategy for constraint selection (default 'uniform').
    """

    def __init__(
        self,
        num_costs: int = 5,
        sim_threshold: float = 0.8,
        conflict_threshold: float = 0.999,
        sampling: str = 'uniform',
    ):
        self.num_costs = num_costs
        self.sim_threshold = sim_threshold
        self.conflict_threshold = conflict_threshold
        assert sampling in ('lambda', 'lambda_normalized', 'uniform'), \
            f"Invalid sampling strategy: {sampling}. Use 'lambda', 'lambda_normalized', or 'uniform'"
        self.sampling = sampling

    def select(
        self,
        cost_grads: list[torch.Tensor],
        lambdas: list[float],
        cost_limits: list[float],
    ) -> tuple[int, float, dict]:
        """Select a non-conflicting constraint gradient via GradS.

        Args:
            cost_grads: List of N flattened gradient vectors, each shape (D,).
            lambdas: List of N per-constraint Lagrange multiplier values.
            cost_limits: List of N per-constraint cost limits.

        Returns:
            (selected_index, scale, diagnostics): Index into cost_grads,
            |G|/N scale factor, and dict with cosine sim matrix + candidate info.
        """
        n = len(cost_grads)
        assert n == self.num_costs

        # 1. Pre-scale gradients by lambda (paper Alg 1: gᵢ = λᵢ·∇Vcᵢ)
        #    Cosine similarity is computed on lambda-scaled gradients.
        #    When λᵢ=0 (not violated), gᵢ=0 → neutral in candidate set.
        G = np.stack([
            lambdas[i] * g.detach().cpu().numpy()
            for i, g in enumerate(cost_grads)
        ])  # (N, D)
        norms = np.linalg.norm(G, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        G_normed = G / norms
        cos_matrix = G_normed @ G_normed.T  # (N, N)

        # 2. Random permutation (paper convention — ensures all constraints
        #    get fair chance as the anchor of the candidate set)
        perm = np.random.permutation(n).tolist()

        # 3. Build candidate set using pre-computed matrix
        candidate_set = [perm[0]]
        for idx in perm[1:]:
            sims = [cos_matrix[idx, j] for j in candidate_set]
            max_sim = max(sims)
            min_sim = min(sims)
            if max_sim < self.sim_threshold and min_sim > -self.conflict_threshold:
                candidate_set.append(idx)

        # 4. Sample from candidate set based on chosen strategy
        if self.sampling == 'uniform':
            probs = np.ones(len(candidate_set)) / len(candidate_set)
        elif self.sampling == 'lambda':
            # softmax(lambda_i) — biased toward highest lambda (most-violated)
            norm_lags = np.array(
                [lambdas[i] for i in candidate_set], dtype=np.float64,
            )
            norm_lags -= norm_lags.max()  # numerical stability
            probs = np.exp(norm_lags)
            total = probs.sum()
            if total < 1e-10:
                probs = np.ones(len(candidate_set)) / len(candidate_set)
            else:
                probs /= total
        else:  # 'lambda_normalized'
            # softmax(lambda_i / cost_limit_i) — v2 behavior
            norm_lags = np.array(
                [lambdas[i] / max(cost_limits[i], 1e-8) for i in candidate_set],
                dtype=np.float64,
            )
            norm_lags -= norm_lags.max()
            probs = np.exp(norm_lags)
            total = probs.sum()
            if total < 1e-10:
                probs = np.ones(len(candidate_set)) / len(candidate_set)
            else:
                probs /= total

        # 5. Sample one constraint
        chosen_pos = np.random.choice(len(candidate_set), p=probs)
        selected_idx = candidate_set[chosen_pos]

        # 6. Scale factor: |G| / N
        scale = len(candidate_set) / n

        # 7. Diagnostics for logging
        diagnostics = {
            'cos_matrix': cos_matrix,         # (N, N) numpy
            'candidate_set': candidate_set,   # list of ints
            'anchor': perm[0],                # which constraint was anchor
        }

        return selected_idx, scale, diagnostics
