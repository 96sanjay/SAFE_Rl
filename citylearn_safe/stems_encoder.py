"""
STEMS Spatial-Temporal Encoder for CityLearn V2G Environment
=============================================================

Implements the GCN-Transformer fusion architecture from:
  "STEMS: Spatial-Temporal Enhanced Safe Multi-Agent Coordination
   for Building Energy Management" (Zhang et al., 2025)

Adapted for centralised OmniSafe agent controlling 17 buildings + 8 EV chargers.

Key improvements over the base implementation:
  1. Observation embedding MLP (paper Fig. 2, first stage)
  2. Learnable adaptive graph construction (Eq. 11 + learned refinement)
  3. Residual GCN blocks with LayerNorm (stabilises RL training)
  4. Full Transformer block with sinusoidal positional encoding (Eq. 13-14)
  5. Gated spatial-temporal fusion (generalises Eq. 15)
  6. Integrated history buffer for temporal window management
  7. Proper batch-dimension handling for vectorised OmniSafe rollouts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse, add_self_loops
import math
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Sinusoidal Positional Encoding  (standard Transformer PE)
# ---------------------------------------------------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal PE for the temporal window positions."""

    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        # Shape: [1, max_len, d_model] — broadcastable over batch & nodes
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., seq_len, d_model]"""
        return x + self.pe[:, : x.size(-2), :]


# ---------------------------------------------------------------------------
# 2. Observation Embedding  (paper Fig. 2 — first block after raw states)
# ---------------------------------------------------------------------------
class ObservationEmbedding(nn.Module):
    """
    Projects heterogeneous per-node raw features into a uniform embedding.
    Two-layer MLP with LayerNorm — matches the 'Observation Embedding'
    block in the STEMS framework diagram.
    """

    def __init__(self, raw_dim: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(raw_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# 3. Adaptive Graph Construction  (Eq. 11 + learned refinement)
# ---------------------------------------------------------------------------
class AdaptiveGraphConstructor(nn.Module):
    """
    Builds edge weights from:
      (a) Prior knowledge  — geographic distance & attribute similarity (Eq. 11)
      (b) Learned attention — node-pair bilinear scoring

    The final weight is a convex combination controlled by a learnable gate,
    so the model can fall back to pure priors early in training and shift
    toward learned structure as it gains experience.
    """

    def __init__(
        self,
        num_nodes: int,
        embed_dim: int,
        sigma_d: float = 1.0,
        sigma_f: float = 1.0,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.sigma_d = sigma_d
        self.sigma_f = sigma_f

        # Learnable node keys for attention-based edge scoring
        self.node_key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.node_query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scale = embed_dim ** -0.5

        # Learnable gate: 0 → pure prior, 1 → pure learned
        self.gate_logit = nn.Parameter(torch.tensor(0.0))

        # Prior weight parameters (Eq. 11 α, β)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def _prior_weights(
        self, dist_matrix: torch.Tensor, attr_diff_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Eq. 11: w_ij = α·exp(−d²/2σ_d²) + β·exp(−‖f‖²/2σ_f²)"""
        alpha = torch.sigmoid(self.alpha)  # keep in [0,1]
        beta = torch.sigmoid(self.beta)
        geo = alpha * torch.exp(-dist_matrix.pow(2) / (2 * self.sigma_d ** 2))
        attr = beta * torch.exp(-attr_diff_matrix.pow(2) / (2 * self.sigma_f ** 2))
        return geo + attr

    def _learned_weights(self, node_embeds: torch.Tensor) -> torch.Tensor:
        """Bilinear attention: softmax(Q K^T / sqrt(d))"""
        Q = self.node_query(node_embeds)  # [B, N, D]
        K = self.node_key(node_embeds)    # [B, N, D]
        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # [B, N, N]
        return torch.softmax(scores, dim=-1)

    def forward(
        self,
        node_embeds: torch.Tensor,
        dist_matrix: torch.Tensor,
        attr_diff_matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            edge_index : [2, E]  (COO format, includes self-loops)
            edge_weight: [E]
            adj_matrix : [B, N, N]  (dense, for inspection / visualisation)
        """
        B = node_embeds.size(0)
        gate = torch.sigmoid(self.gate_logit)

        prior = self._prior_weights(dist_matrix, attr_diff_matrix)  # [N, N]
        prior = prior.unsqueeze(0).expand(B, -1, -1)                # [B, N, N]

        learned = self._learned_weights(node_embeds)                # [B, N, N]

        adj = (1 - gate) * prior + gate * learned  # [B, N, N]

        # For GCNConv we need a single graph — average over batch
        adj_mean = adj.mean(dim=0)  # [N, N]
        # Threshold tiny weights to keep graph sparse
        adj_mean = adj_mean * (adj_mean > 0.01).float()
        edge_index, edge_weight = dense_to_sparse(adj_mean)
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, fill_value=1.0, num_nodes=self.num_nodes
        )

        return edge_index, edge_weight, adj


# ---------------------------------------------------------------------------
# 4. Residual GCN Block  (Eq. 12 with skip connections + LayerNorm)
# ---------------------------------------------------------------------------
class ResidualGCNBlock(nn.Module):
    """
    Single GCN layer wrapped with:
      • Pre-LayerNorm (stabilises gradients in deep RL)
      • Residual connection (if dims match)
      • Dropout
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.conv = GCNConv(in_dim, out_dim, improved=True, add_self_loops=False)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ) -> torch.Tensor:
        """x: [N, D]  (single graph, not batched)"""
        h = self.norm(x)
        h = self.conv(h, edge_index, edge_weight=edge_weight)
        h = self.act(h)
        h = self.drop(h)
        return h + self.skip(x)


class SpatialEncoder(nn.Module):
    """Stack of ResidualGCN blocks — implements multi-layer Eq. 12."""

    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [ResidualGCNBlock(embed_dim, embed_dim, dropout) for _ in range(num_layers)]
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
        return x


# ---------------------------------------------------------------------------
# 5. Temporal Transformer Block  (Eqs. 13-14, enhanced)
# ---------------------------------------------------------------------------
class TemporalTransformerBlock(nn.Module):
    """
    Full Transformer encoder block applied per-node across the temporal window.
    Includes:
      • Multi-head self-attention (Eq. 13)
      • Weighted aggregation via values (Eq. 14)
      • Feed-forward sublayer
      • Pre-norm residual pattern
      • Sinusoidal positional encoding
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
        max_window: int = 64,
    ):
        super().__init__()
        self.pos_enc = SinusoidalPositionalEncoding(embed_dim, max_len=max_window)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * ff_mult, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        seq: [B*N, T, D]  — temporal sequence per node
        Returns:
            out       : [B*N, D]  — last-position output (current step representation)
            attn_weights: [B*N, T, T]
        """
        # Add positional encoding
        seq = self.pos_enc(seq)

        # Self-attention sublayer
        h = self.norm1(seq)
        attn_out, attn_weights = self.attn(h, h, h)
        seq = seq + attn_out

        # Feed-forward sublayer
        h = self.norm2(seq)
        seq = seq + self.ff(h)

        # Take the last position as the temporal summary for current step
        out = seq[:, -1, :]  # [B*N, D]
        return out, attn_weights


# ---------------------------------------------------------------------------
# 6. Gated Spatial-Temporal Fusion  (generalised Eq. 15)
# ---------------------------------------------------------------------------
class GatedFusion(nn.Module):
    """
    Eq. 15: r_t = W_s · h_spatial + W_t · z_temporal + b
    Enhanced with a learnable sigmoid gate so the model can dynamically
    weight spatial vs temporal features per node.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W_s = nn.Linear(hidden_dim, hidden_dim)
        self.W_t = nn.Linear(hidden_dim, hidden_dim)
        # Gate: takes concatenation of both, outputs per-dimension weight
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, h_spatial: torch.Tensor, z_temporal: torch.Tensor
    ) -> torch.Tensor:
        """Both inputs: [B, N, D] or [N, D]"""
        s = self.W_s(h_spatial)
        t = self.W_t(z_temporal)
        g = self.gate(torch.cat([h_spatial, z_temporal], dim=-1))
        fused = g * s + (1 - g) * t
        return self.norm(fused)


# ---------------------------------------------------------------------------
# 7. History Buffer  (manages sliding window for temporal attention)
# ---------------------------------------------------------------------------
class HistoryBuffer:
    """
    Ring buffer that stores the last `window_size` embedded observations
    per node.  Call .push() every environment step; .get() returns the
    padded temporal window for the Transformer.

    Works with both single-env and vectorised envs (batch dim).
    """

    def __init__(self, window_size: int, num_nodes: int, embed_dim: int):
        self.window_size = window_size
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.buffer = None  # lazily initialised on first push
        self.count = 0

    def reset(self, batch_size: int = 1, device: torch.device = torch.device("cpu")):
        self.buffer = torch.zeros(
            batch_size, self.num_nodes, self.window_size, self.embed_dim,
            device=device,
        )
        self.count = 0

    def push(self, node_embeds: torch.Tensor):
        """
        node_embeds: [B, N, D]
        Shifts buffer left by 1 and appends new observation at the end.
        """
        if self.buffer is None:
            B = node_embeds.size(0)
            self.reset(B, node_embeds.device)

        self.buffer = torch.roll(self.buffer, shifts=-1, dims=2)
        self.buffer[:, :, -1, :] = node_embeds
        self.count = min(self.count + 1, self.window_size)

    def get(self) -> torch.Tensor:
        """Returns [B, N, T, D] — zero-padded if fewer than window_size steps."""
        return self.buffer.clone()


# ---------------------------------------------------------------------------
# 8. Full STEMS Encoder  — main class
# ---------------------------------------------------------------------------
class STEMSEncoder(nn.Module):
    """
    Complete spatial-temporal encoder following the STEMS paper architecture
    (Fig. 2), adapted for a centralised OmniSafe agent in CityLearn.

    Input:  flat observation vector from CityLearnSafetyEnvV3
    Output: flat feature vector to feed into OmniSafe's actor & critic MLPs

    Architecture pipeline:
        raw obs → reshape to per-node features
                → ObservationEmbedding
                → push to HistoryBuffer
                → AdaptiveGraphConstructor → edge_index, edge_weight
                → SpatialEncoder (multi-layer GCN)
                → TemporalTransformerBlock (over history window)
                → GatedFusion
                → flatten → output projection
    """

    def __init__(
        self,
        obs_dim: int,
        num_buildings: int = 17,
        num_evs: int = 8,
        hidden_dim: int = 64,
        num_gcn_layers: int = 3,
        num_heads: int = 4,
        temporal_window: int = 24,
        ff_mult: int = 4,
        dropout: float = 0.1,
        output_dim: Optional[int] = None,
        # Prior graph information
        building_distances: Optional[torch.Tensor] = None,
        building_attr_diffs: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            obs_dim:       Total flat observation size from env.observation_space.
            num_buildings: Number of building nodes in the graph (17).
            num_evs:       Number of EV chargers (8) — EV features are associated
                           with their host buildings, not separate nodes.
            hidden_dim:    Embedding / hidden dimension throughout.
            num_gcn_layers: Number of stacked GCN layers.
            num_heads:     Attention heads in temporal Transformer.
            temporal_window: History window T for temporal attention.
            ff_mult:       Feed-forward expansion factor in Transformer.
            dropout:       Dropout rate.
            output_dim:    Final output vector size. If None, defaults to
                           num_buildings * hidden_dim (flattened node features).
            building_distances:  [N, N] pairwise distance matrix (prior).
            building_attr_diffs: [N, N] pairwise attribute difference matrix (prior).
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.num_buildings = num_buildings
        self.num_nodes = num_buildings  # graph nodes = buildings
        self.hidden_dim = hidden_dim
        self.temporal_window = temporal_window

        # Compute per-node feature dimension from flat observation
        # The observation contains global features + per-building features
        # We split: first estimate features_per_node
        self.features_per_node = obs_dim // num_buildings
        self.remainder = obs_dim - (self.features_per_node * num_buildings)

        # If there are remainder features (global context), we append them to each node
        raw_node_dim = self.features_per_node + (
            self.remainder if self.remainder > 0 else 0
        )

        # --- Architecture blocks ---
        self.obs_embed = ObservationEmbedding(raw_node_dim, hidden_dim, dropout)

        self.graph_constructor = AdaptiveGraphConstructor(
            num_nodes=num_buildings,
            embed_dim=hidden_dim,
        )

        self.spatial_encoder = SpatialEncoder(
            embed_dim=hidden_dim,
            num_layers=num_gcn_layers,
            dropout=dropout,
        )

        self.temporal_encoder = TemporalTransformerBlock(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            ff_mult=ff_mult,
            dropout=dropout,
            max_window=temporal_window + 8,
        )

        self.fusion = GatedFusion(hidden_dim)

        # Output projection: flatten node features → fixed-size vector
        _out_dim = output_dim if output_dim is not None else (num_buildings * hidden_dim)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(num_buildings * hidden_dim),
            nn.Linear(num_buildings * hidden_dim, _out_dim),
            nn.ReLU(),
        )
        self.output_dim = _out_dim

        # History buffer (created per reset)
        self.history = HistoryBuffer(temporal_window, num_buildings, hidden_dim)

        # Register prior graph matrices (can be None — then pure learned graph)
        if building_distances is not None:
            self.register_buffer("dist_matrix", building_distances)
        else:
            # Default: uniform small distance — graph constructor learns structure
            self.register_buffer(
                "dist_matrix",
                torch.ones(num_buildings, num_buildings) * 0.5,
            )

        if building_attr_diffs is not None:
            self.register_buffer("attr_diff_matrix", building_attr_diffs)
        else:
            self.register_buffer(
                "attr_diff_matrix",
                torch.ones(num_buildings, num_buildings) * 0.5,
            )

    def _reshape_obs_to_nodes(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Splits a flat observation [B, obs_dim] into per-node features [B, N, F].
        If obs_dim is not evenly divisible by num_buildings, the remainder
        features (global context) are broadcast-appended to every node.
        """
        B = obs.size(0)
        N = self.num_buildings
        F_per = self.features_per_node

        # Per-building features
        node_feats = obs[:, : N * F_per].reshape(B, N, F_per)

        if self.remainder > 0:
            global_feats = obs[:, N * F_per :]  # [B, remainder]
            global_feats = global_feats.unsqueeze(1).expand(B, N, -1)  # [B, N, R]
            node_feats = torch.cat([node_feats, global_feats], dim=-1)  # [B, N, F+R]

        return node_feats

    def reset_history(self, batch_size: int = 1, device: torch.device = None):
        """Call at the start of each episode."""
        dev = device or next(self.parameters()).device
        self.history.reset(batch_size, dev)

    def forward(
        self,
        obs: torch.Tensor,
        push_history: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            obs: [B, obs_dim]  or  [obs_dim]  — flat observation from env.
            push_history: If True, appends current embedding to history buffer.
                          Set False during critic-only forward passes if you
                          don't want to double-count steps.

        Returns:
            features: [B, output_dim]  — ready to feed into OmniSafe MLP.
        """
        # Handle unbatched input
        squeeze = False
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            squeeze = True

        B = obs.size(0)
        N = self.num_nodes

        # ---- 1. Reshape & embed per-node features ----
        node_raw = self._reshape_obs_to_nodes(obs)        # [B, N, raw_dim]
        node_embed = self.obs_embed(node_raw)              # [B, N, D]

        # ---- 2. Push to history buffer ----
        if push_history:
            self.history.push(node_embed.detach())  # detach to avoid backprop through buffer

        # ---- 3. Adaptive graph construction ----
        edge_index, edge_weight, adj = self.graph_constructor(
            node_embed, self.dist_matrix, self.attr_diff_matrix
        )

        # ---- 4. Spatial encoding (GCN over buildings) ----
        # GCNConv expects [N, D] per sample — process batch in a loop
        # (for vector_env_nums=1 this is just 1 iteration)
        h_spatial_list = []
        for b in range(B):
            h_b = self.spatial_encoder(node_embed[b], edge_index, edge_weight)
            h_spatial_list.append(h_b)
        h_spatial = torch.stack(h_spatial_list, dim=0)  # [B, N, D]

        # ---- 5. Temporal encoding (Transformer over history window) ----
        history_seq = self.history.get()  # [B, N, T, D]

        # Reshape for per-node temporal attention: [B*N, T, D]
        BN = B * N
        hist_flat = history_seq.reshape(BN, self.temporal_window, self.hidden_dim)

        z_temporal, attn_weights = self.temporal_encoder(hist_flat)  # [B*N, D]
        z_temporal = z_temporal.reshape(B, N, self.hidden_dim)       # [B, N, D]

        # ---- 6. Gated fusion ----
        r = self.fusion(h_spatial, z_temporal)  # [B, N, D]

        # ---- 7. Flatten & project ----
        r_flat = r.reshape(B, N * self.hidden_dim)  # [B, N*D]
        features = self.output_proj(r_flat)          # [B, output_dim]

        if squeeze:
            features = features.squeeze(0)

        return features


# ---------------------------------------------------------------------------
# 9. OmniSafe Integration Wrapper
# ---------------------------------------------------------------------------
class STEMSActorCriticWrapper(nn.Module):
    """
    Wraps STEMSEncoder so it can be used as a feature extractor in front
    of OmniSafe's standard MLP actor and critic heads.

    Usage in OmniSafe custom model:
        encoder = STEMSActorCriticWrapper(obs_dim=obs_dim, ...)
        # Then configure OmniSafe to use encoder.output_dim as its
        # effective observation dimension.
    """

    def __init__(self, obs_dim: int, **encoder_kwargs):
        super().__init__()
        self.encoder = STEMSEncoder(obs_dim=obs_dim, **encoder_kwargs)
        self.output_dim = self.encoder.output_dim
        self._episode_started = False

    def reset(self, batch_size: int = 1):
        """Call at episode boundaries."""
        self.encoder.reset_history(batch_size)
        self._episode_started = True

    def forward(self, obs: torch.Tensor, push_history: bool = True) -> torch.Tensor:
        # Auto-reset history if buffer is uninitialised
        if self.encoder.history.buffer is None:
            B = obs.size(0) if obs.dim() > 1 else 1
            self.reset(B)
        return self.encoder(obs, push_history=push_history)


# ---------------------------------------------------------------------------
# 10. Convenience: build prior graph matrices from building metadata
# ---------------------------------------------------------------------------
def build_prior_graph(
    building_positions: list[tuple[float, float]],
    building_types: list[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Constructs the distance and attribute-difference matrices from
    building metadata for use as priors in AdaptiveGraphConstructor.

    Args:
        building_positions: List of (x, y) coordinates for each building.
        building_types:     List of type strings, e.g. ['residential', 'commercial', ...].

    Returns:
        dist_matrix:     [N, N] pairwise Euclidean distances (normalised to [0,1]).
        attr_diff_matrix: [N, N] binary: 0 if same type, 1 if different type.
    """
    N = len(building_positions)
    positions = torch.tensor(building_positions, dtype=torch.float32)
    dist_matrix = torch.cdist(positions, positions, p=2)
    # Normalise to [0, 1]
    dmax = dist_matrix.max()
    if dmax > 0:
        dist_matrix = dist_matrix / dmax

    type_set = sorted(set(building_types))
    type_ids = torch.tensor([type_set.index(t) for t in building_types])
    attr_diff_matrix = (type_ids.unsqueeze(0) != type_ids.unsqueeze(1)).float()

    return dist_matrix, attr_diff_matrix


# ---------------------------------------------------------------------------
# 11. Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Simulate CityLearn obs: 17 buildings, each with ~15 features ≈ 255 dims
    # (adjust to your actual obs_dim)
    OBS_DIM = 255
    NUM_BUILDINGS = 17
    HIDDEN = 64
    WINDOW = 24
    BATCH = 1

    encoder = STEMSEncoder(
        obs_dim=OBS_DIM,
        num_buildings=NUM_BUILDINGS,
        hidden_dim=HIDDEN,
        num_gcn_layers=3,
        num_heads=4,
        temporal_window=WINDOW,
        output_dim=256,
    )

    encoder.reset_history(BATCH)

    # Simulate 5 environment steps
    for step in range(5):
        obs = torch.randn(BATCH, OBS_DIM)
        feat = encoder(obs)
        print(f"Step {step}: obs {obs.shape} → features {feat.shape}")

    total_params = sum(p.numel() for p in encoder.parameters())
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable:        {trainable:,}")
    print(f"Output dim:       {encoder.output_dim}")

