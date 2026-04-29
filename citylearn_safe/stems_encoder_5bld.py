"""
STEMS GCN-Transformer Encoder for 5-Building CityLearn V2G
===========================================================

Pure PyTorch reimplementation of the STEMS spatial-temporal encoder
(stems_encoder.py) adapted for the 5-building schema. No torch_geometric
dependency — uses a simple matrix-multiply GCN instead of GCNConv.

Uses ObsIndex from schema_index.py for exact observation parsing
(no naive obs_dim // num_buildings splitting).

Architecture (v2 — stateless, PPO-compatible):
    obs → ObsIndex parse → per-node [B, N, F_node]
        → ObservationEmbedding → [B, N, D]
        → AdaptiveGraphConstructor → adj [N, N]
        → SpatialEncoder (GCN stack) → h_spatial [B, N, D]
        → SpatialSelfAttention (MHA over N nodes) → z_attn [B, N, D]
        → GatedFusion → r [B, N, D]
        → flatten → output_proj → [B, output_dim]

v2 changes from v1:
  - Removed HistoryBuffer + TemporalTransformerBlock (caused rollout/update
    behavioral split breaking PPO importance sampling, detached gradients)
  - Added SpatialSelfAttention (stateless, same function in rollout & update)
  - Removed output_proj final ReLU (was blocking negative features)
  - Fixed GCN double normalization (graph constructor already normalizes)
"""

import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Pure PyTorch GCN Layer (replaces torch_geometric GCNConv)
# ---------------------------------------------------------------------------
class SimpleGCNLayer(nn.Module):
    """
    GCN layer: H' = linear(A_norm @ H)

    Expects pre-normalized adjacency matrix (from AdaptiveGraphConstructor
    which applies softmax + self-loops). No additional symmetric normalization
    to avoid double-normalization feature shrinkage.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x:   [N, D_in]
        adj: [N, N] (pre-normalized from AdaptiveGraphConstructor)
        Returns: [N, D_out]
        """
        return self.linear(adj @ x)


# ---------------------------------------------------------------------------
# 2. Residual GCN Block
# ---------------------------------------------------------------------------
class ResidualGCNBlock(nn.Module):
    """GCN layer + LayerNorm + residual + dropout."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.gcn = SimpleGCNLayer(in_dim, out_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.gcn(h, adj)
        h = self.act(h)
        h = self.drop(h)
        return h + self.skip(x)


class SpatialEncoder(nn.Module):
    """Stack of ResidualGCN blocks."""

    def __init__(self, embed_dim: int, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [ResidualGCNBlock(embed_dim, embed_dim, dropout) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, adj)
        return x


# ---------------------------------------------------------------------------
# 3. Spatial Self-Attention (replaces temporal transformer)
# ---------------------------------------------------------------------------
class SpatialSelfAttention(nn.Module):
    """
    Multi-head self-attention over building nodes.

    Stateless — same function in rollout and update, so PPO importance
    sampling ratios are correct. Gets proper gradients during backprop.

    This replaces the TemporalTransformerBlock which used a HistoryBuffer
    that broke PPO by computing different functions in rollout vs update.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, D]  (N = num_buildings)
        Returns: [B, N, D]
        """
        # Pre-norm self-attention
        h = self.norm(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        # Pre-norm FFN
        h = self.norm2(x)
        x = x + self.ff(h)
        return x


# ---------------------------------------------------------------------------
# 4. Gated Spatial-Attention Fusion
# ---------------------------------------------------------------------------
class GatedFusion(nn.Module):
    """Learnable gate combining spatial (GCN) and attention/temporal features per node.

    R26g: Added gate monitoring — stores last gate values for diagnostic logging.
    If gate_mean → 1.0, the temporal signal is being suppressed.
    If gate_mean → 0.0, the spatial signal is being suppressed.
    Healthy range: 0.3–0.7 (both streams contribute).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W_s = nn.Linear(hidden_dim, hidden_dim)
        self.W_t = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        # R26g: gate monitoring (detached, no grad impact)
        self._last_gate_mean: float = 0.5
        self._last_gate_std: float = 0.0

    def forward(self, h_spatial: torch.Tensor, z_attn: torch.Tensor) -> torch.Tensor:
        s = self.W_s(h_spatial)
        t = self.W_t(z_attn)
        g = self.gate(torch.cat([h_spatial, z_attn], dim=-1))
        # R26g: store gate stats for logging (detached, no memory leak)
        with torch.no_grad():
            self._last_gate_mean = g.mean().item()
            self._last_gate_std = g.std().item()
        return self.norm(g * s + (1 - g) * t)

    def gate_stats(self) -> dict:
        """Return last gate statistics for diagnostic logging."""
        return {
            "gate_mean": self._last_gate_mean,
            "gate_std": self._last_gate_std,
        }


# ---------------------------------------------------------------------------
# 5. Observation Embedding
# ---------------------------------------------------------------------------
class ObservationEmbedding(nn.Module):
    """Two-layer MLP with LayerNorm to project per-node raw features to uniform embedding."""

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
# 6. Adaptive Graph Constructor (pure learned, no prior matrices)
# ---------------------------------------------------------------------------
class AdaptiveGraphConstructor(nn.Module):
    """
    Builds adjacency matrix from node embeddings via bilinear attention.
    Output is row-normalized (softmax) with self-loops.

    NOTE: Output is already normalized — downstream GCN layers should NOT
    apply additional symmetric normalization to avoid feature shrinkage.
    """

    def __init__(self, num_nodes: int, embed_dim: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.node_query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scale = embed_dim ** -0.5
        # Gate: 0 = uniform prior, 1 = fully learned
        self.gate_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, node_embeds: torch.Tensor) -> torch.Tensor:
        """
        node_embeds: [B, N, D]
        Returns: adj [N, N] row-normalized with self-loops (averaged over batch)
        """
        B, N, _ = node_embeds.shape
        gate = torch.sigmoid(self.gate_logit)

        # Uniform prior: all-ones (after normalization = uniform weights)
        prior = torch.ones(N, N, device=node_embeds.device) / N

        # Learned attention scores
        Q = self.node_query(node_embeds)  # [B, N, D]
        K = self.node_key(node_embeds)    # [B, N, D]
        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # [B, N, N]
        learned = torch.softmax(scores, dim=-1).mean(dim=0)  # [N, N]

        # Mix prior and learned
        adj = (1 - gate) * prior + gate * learned

        # Add self-loops and re-normalize rows to sum to 1
        adj = adj + torch.eye(N, device=adj.device)
        adj = adj / adj.sum(dim=-1, keepdim=True)

        return adj


# ---------------------------------------------------------------------------
# 7. Observation Parser (ObsIndex-based, replaces naive splitting)
# ---------------------------------------------------------------------------
def build_node_indices(obs_index, num_buildings: int) -> Dict:
    """
    Extract observation indices organized by building node.

    Returns dict with:
      - building_indices: list of [4] index lists per building
        (non_shiftable_load, solar_gen, soc, net_consumption)
      - ev_indices: list of [7] index lists per building (or None if no charger)
      - global_indices: list of scalar observation indices (shared features)
      - base_obs_dim: dimension of base obs (before forecast wrapper)
    """
    # Per-building features (4 per building)
    building_indices = []
    for i in range(num_buildings):
        building_indices.append([
            obs_index.non_shiftable_load[i],
            obs_index.solar_generation[i],
            obs_index.electrical_storage_soc[i],
            obs_index.net_electricity_consumption[i],
        ])

    # Map charger IDs to building indices
    ev_feature_names = [
        "connected_state", "departure_time", "required_soc_departure",
        "soc", "battery_capacity", "incoming_state", "estimated_arrival_time",
    ]

    # Parse charger building numbers
    charger_to_building_idx = {}
    for charger_id in sorted(obs_index.ev.keys()):
        match = re.match(r"charger_(\d+)_(\d+)", charger_id)
        if match:
            bld_num = int(match.group(1))
            bld_idx = bld_num - 1
            if bld_idx < num_buildings:
                charger_to_building_idx[charger_id] = bld_idx

    ev_indices: List[Optional[List[int]]] = [None] * num_buildings
    for charger_id, bld_idx in charger_to_building_idx.items():
        feats = obs_index.ev[charger_id]
        ev_indices[bld_idx] = [feats[f] for f in ev_feature_names]

    # Global (shared) indices
    global_indices = [
        obs_index.month_cos, obs_index.month_sin,
        obs_index.day_type_cos, obs_index.day_type_sin,
        obs_index.hour_cos, obs_index.hour_sin,
        obs_index.carbon_intensity,
        obs_index.electricity_pricing,
        obs_index.electricity_pricing_predicted_1,
        obs_index.electricity_pricing_predicted_2,
        obs_index.electricity_pricing_predicted_3,
    ]
    if obs_index.washing_machine_1_start_time_step is not None:
        global_indices.append(obs_index.washing_machine_1_start_time_step)
    if obs_index.washing_machine_1_end_time_step is not None:
        global_indices.append(obs_index.washing_machine_1_end_time_step)

    return dict(
        building_indices=building_indices,  # List[List[int]], shape [N][4]
        ev_indices=ev_indices,              # List[Optional[List[int]]], shape [N][7 or None]
        global_indices=global_indices,      # List[int]
        base_obs_dim=obs_index.obs_dim,     # int
    )


# ---------------------------------------------------------------------------
# 8. Main STEMS Encoder for 5 Buildings (v2)
# ---------------------------------------------------------------------------
class STEMSEncoder5Bld(nn.Module):
    """
    STEMS GCN + Self-Attention encoder adapted for N-building CityLearn.

    Uses ObsIndex for exact observation parsing. Pure PyTorch GCN.
    Designed as a drop-in mean_net for OmniSafe's GaussianLearningActor.

    v2: Stateless architecture — no HistoryBuffer or temporal transformer.
    Uses SpatialSelfAttention over building nodes instead. Same function
    computed in rollout and update, so PPO importance sampling is correct.
    """

    def __init__(
        self,
        obs_dim: int,
        node_info: Dict,
        num_buildings: int = 5,
        hidden_dim: int = 64,
        global_hidden: int = 32,
        num_gcn_layers: int = 3,
        num_heads: int = 4,
        temporal_window: int = 24,  # kept for API compat, not used
        ff_mult: int = 4,
        dropout: float = 0.1,
        output_dim: int = 256,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_buildings = num_buildings
        self.hidden_dim = hidden_dim
        self.base_obs_dim = node_info["base_obs_dim"]

        # Store index tensors as buffers for fast gathering
        bld_idx = node_info["building_indices"]  # [N][4]
        self.register_buffer("bld_idx", torch.tensor(bld_idx, dtype=torch.long))  # [N, 4]

        # EV indices: [N, 7], with -1 for buildings without chargers
        ev_idx_list = []
        ev_mask_list = []
        for ev in node_info["ev_indices"]:
            if ev is not None:
                ev_idx_list.append(ev)
                ev_mask_list.append(1.0)
            else:
                ev_idx_list.append([0] * 7)  # dummy indices (masked out)
                ev_mask_list.append(0.0)
        self.register_buffer("ev_idx", torch.tensor(ev_idx_list, dtype=torch.long))  # [N, 7]
        self.register_buffer("ev_mask", torch.tensor(ev_mask_list, dtype=torch.float32))  # [N]

        self.register_buffer(
            "global_idx", torch.tensor(node_info["global_indices"], dtype=torch.long)
        )

        base_obs_dim = node_info["base_obs_dim"]
        forecast_dim = obs_dim - base_obs_dim  # extra dims from ForecastObsWrapper
        global_raw_dim = len(node_info["global_indices"]) + forecast_dim

        # --- Architecture ---

        # Global encoder: shared features + forecast → compact embedding
        self.global_enc = nn.Sequential(
            nn.Linear(global_raw_dim, 64),
            nn.ReLU(),
            nn.Linear(64, global_hidden),
            nn.LayerNorm(global_hidden),
            nn.ReLU(),
        )

        # Per-node raw dim: building(4) + ev(7) + global_emb(global_hidden)
        node_raw_dim = 4 + 7 + global_hidden
        self.obs_embed = ObservationEmbedding(node_raw_dim, hidden_dim, dropout)

        self.graph_constructor = AdaptiveGraphConstructor(num_buildings, hidden_dim)

        self.spatial_encoder = SpatialEncoder(hidden_dim, num_gcn_layers, dropout)

        # Spatial self-attention over building nodes (replaces temporal transformer)
        self.spatial_attn = SpatialSelfAttention(hidden_dim, num_heads, dropout)

        self.fusion = GatedFusion(hidden_dim)

        # Output: flatten all node features → output_dim (NO final ReLU)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(num_buildings * hidden_dim),
            nn.Linear(num_buildings * hidden_dim, output_dim),
        )
        self.output_dim = output_dim

    def _parse_obs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parse flat obs into per-building and global features using ObsIndex.

        Returns:
            node_raw: [B, N, 4+7] per-building + EV features
            global_raw: [B, G] global + forecast features
        """
        B = obs.size(0)
        N = self.num_buildings

        # Building features: [B, N, 4]
        bld_feats = obs[:, self.bld_idx.view(-1)].view(B, N, 4)

        # EV features: [B, N, 7] (masked to zero for buildings without chargers)
        ev_feats = obs[:, self.ev_idx.view(-1)].view(B, N, 7)
        ev_feats = ev_feats * self.ev_mask.view(1, N, 1)  # zero out dummy indices

        node_raw = torch.cat([bld_feats, ev_feats], dim=-1)  # [B, N, 11]

        # Global features: shared scalars from base obs + forecast dims
        global_shared = obs[:, self.global_idx]  # [B, len(global_idx)]

        # Forecast features start right after the base observation
        if obs.size(1) > self.base_obs_dim:
            forecast_feats = obs[:, self.base_obs_dim:]  # [B, forecast_dim]
            global_raw = torch.cat([global_shared, forecast_feats], dim=-1)
        else:
            global_raw = global_shared

        return node_raw, global_raw

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: [B, obs_dim] or [obs_dim]
        Returns: [B, output_dim] feature vector for OmniSafe actor/critic.

        Fully stateless — same computation in rollout and PPO update.
        """
        squeeze = False
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            squeeze = True

        B = obs.size(0)
        N = self.num_buildings

        # 1. Parse observations
        node_raw, global_raw = self._parse_obs(obs)  # [B,N,11], [B,G]

        # 2. Encode global context and broadcast to nodes
        global_emb = self.global_enc(global_raw)  # [B, global_hidden]
        global_broadcast = global_emb.unsqueeze(1).expand(B, N, -1)  # [B, N, global_hidden]

        # 3. Concatenate per-node features with global context
        node_input = torch.cat([node_raw, global_broadcast], dim=-1)  # [B, N, 11+global_hidden]

        # 4. Embed to hidden dim
        node_embed = self.obs_embed(node_input)  # [B, N, D]

        # 5. Adaptive graph
        adj = self.graph_constructor(node_embed)  # [N, N]

        # 6. Spatial GCN (process batch in loop for per-sample adj compat)
        h_spatial_list = []
        for b in range(B):
            h_b = self.spatial_encoder(node_embed[b], adj)  # [N, D]
            h_spatial_list.append(h_b)
        h_spatial = torch.stack(h_spatial_list, dim=0)  # [B, N, D]

        # 7. Spatial self-attention over building nodes
        z_attn = self.spatial_attn(h_spatial)  # [B, N, D]

        # 8. Gated fusion (GCN spatial + attention)
        r = self.fusion(h_spatial, z_attn)  # [B, N, D]

        # 9. Flatten and project (no ReLU — allow negative features)
        r_flat = r.reshape(B, N * self.hidden_dim)  # [B, N*D]
        features = self.output_proj(r_flat)  # [B, output_dim]

        if squeeze:
            features = features.squeeze(0)

        return features


# ---------------------------------------------------------------------------
# 9. Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Simulate 5-building obs (no real env needed for shape test)
    OBS_DIM = 198  # base(~70) + forecast(128)
    NUM_BUILDINGS = 5

    # Mock node_info (would come from build_node_indices in real usage)
    mock_node_info = dict(
        building_indices=[[i*4+j for j in range(4)] for i in range(5)],
        ev_indices=[[20+i*7+j for j in range(7)] if i in [0, 3, 4] else None for i in range(5)],
        global_indices=list(range(41, 54)),  # 13 global features
        base_obs_dim=70,
    )

    encoder = STEMSEncoder5Bld(
        obs_dim=OBS_DIM,
        node_info=mock_node_info,
        num_buildings=NUM_BUILDINGS,
        hidden_dim=64,
        output_dim=256,
    )

    # Test forward pass (no history needed — stateless)
    for step in range(5):
        obs = torch.randn(1, OBS_DIM)
        feat = encoder(obs)
        print(f"Step {step}: obs {obs.shape} -> features {feat.shape}")

    # Test with gradient (simulates PPO update)
    obs = torch.randn(4, OBS_DIM)
    obs.requires_grad_(True)
    feat = encoder(obs)
    loss = feat.sum()
    loss.backward()
    print(f"\nGrad test: obs {obs.shape} -> features {feat.shape}, grad norm: {obs.grad.norm():.4f}")

    total = sum(p.numel() for p in encoder.parameters())
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total:,}")
    print(f"Trainable:        {trainable:,}")
    print(f"Output dim:       {encoder.output_dim}")
