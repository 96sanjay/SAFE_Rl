"""
STEMS Encoder V3: Per-Node Temporal Transformer + Spatial GCN
=============================================================

Extends v2 (stems_encoder_5bld.py) by adding per-building temporal modeling.
The key insight: temporal history is now embedded in the observation vector
by a TemporalHistoryWrapper, so the encoder is still STATELESS (no internal
buffers that break PPO importance sampling).

Architecture (R26g: forecast-augmented temporal):
    obs [B, obs_dim] where obs_dim = current_obs(198) + T * features_per_step
      |
      +-- current obs -> parse via ObsIndex -> per-node [B, N, 11] + global
      |     forecast split: price/load/solar [B, 24, 3] -> temporal transformer
      |                     ev_urgency + time_enc [B, 56] -> global_enc
      +-- history tail -> reshape [B, T, features_per_step]
      |     per-node: extract [soc_i, net_consumption_i, price] -> [B, T, K]
      |
      v
    Per-Node Temporal Transformer (forecast-augmented in R26g)
      For each building i:
        backward: history_i [B, T_back, K] -> input_proj -> [B, T_back, 32]
        forward:  forecast  [B, T_fwd, 3]  -> forecast_proj -> [B, T_fwd, 32]
        concat -> [B, T_back+T_fwd, 32] -> + sinusoidal PE
        -> TransformerEncoderLayer -> last position -> [B, 32]
        -> Linear(32, 64)
      Stack -> z_temporal [B, N, 64]
      |
      v
    Current Node Embedding (reused from v2)
      node_input [B, N, 43] -> ObservationEmbedding -> [B, N, 64]
      |
      v
    Combine: concat(node_embed, z_temporal) -> Linear -> hidden_dim
      |
      v
    Spatial GCN (reused from v2)
      -> AdaptiveGraphConstructor -> SpatialEncoder -> h_spatial [B, N, 64]
      |
      v
    GatedFusion(h_spatial, z_temporal) -> r [B, N, 64]
      |
      v
    Flatten -> Linear -> [B, 256]

Reuses from v2: SimpleGCNLayer, ResidualGCNBlock, SpatialEncoder,
    AdaptiveGraphConstructor, GatedFusion, ObservationEmbedding, build_node_indices
"""

import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from citylearn_safe.stems_encoder_5bld import (
    AdaptiveGraphConstructor,
    GatedFusion,
    ObservationEmbedding,
    SpatialEncoder,
    build_node_indices,
)


# ---------------------------------------------------------------------------
# 1. Sinusoidal Positional Encoding
# ---------------------------------------------------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for transformer inputs.
    Pre-computes encodings for up to max_len positions.
    """

    def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )  # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer: [1, max_len, d_model] for easy broadcasting
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, d_model]
        Returns: [B, T, d_model] with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# 2. Per-Node Temporal Transformer
# ---------------------------------------------------------------------------
class PerNodeTemporalTransformer(nn.Module):
    """
    Processes per-building temporal history independently via a shared
    Transformer encoder. Each building attends to its OWN T-step history.

    R26g: Optionally appends forecast positions (price/load/solar) to the
    temporal sequence so the transformer can attend across past AND future.

    Stateless: all history comes from the observation vector, not from
    internal buffers. Safe for PPO importance sampling.
    """

    def __init__(
        self,
        input_features: int = 3,  # soc_i, net_consumption_i, price
        temporal_hidden: int = 32,
        temporal_heads: int = 4,
        ff_dim: int = 128,
        dropout: float = 0.1,
        max_len: int = 128,
        forecast_features: int = 0,  # R26g: 3 (price, load, solar) or 0 to disable
        num_layers: int = 2,  # R26g: 2 layers for compositional temporal reasoning
        pool_mode: str = "mean",  # R26g: "mean", "last", or "cls"
    ):
        super().__init__()
        self.pool_mode = pool_mode
        self.input_proj = nn.Linear(input_features, temporal_hidden)
        self.pos_enc = SinusoidalPositionalEncoding(temporal_hidden, max_len, dropout)

        # R26g: separate projection for forecast features (different dim than history)
        self.forecast_features = forecast_features
        if forecast_features > 0:
            self.forecast_proj = nn.Linear(forecast_features, temporal_hidden)

        # R26g: optional CLS token for aggregation
        if pool_mode == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, temporal_hidden) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=temporal_hidden,
            nhead=temporal_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        # R26g: 2 layers enables compositional temporal reasoning
        # (e.g., "SoC low AND departure soon AND price high")
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(temporal_hidden)

    def forward(self, x: torch.Tensor, forecast_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, T_back, input_features]  (temporal history for ONE building)
        forecast_seq: [B, T_fwd, forecast_features] (optional future forecast)
        Returns: [B, temporal_hidden]  (summary embedding for that building)
        """
        h = self.input_proj(x)       # [B, T_back, temporal_hidden]

        # R26g: append forecast positions to temporal sequence
        if forecast_seq is not None and self.forecast_features > 0:
            f = self.forecast_proj(forecast_seq)  # [B, T_fwd, temporal_hidden]
            h = torch.cat([h, f], dim=1)          # [B, T_back+T_fwd, temporal_hidden]

        # R26g: prepend CLS token if using CLS pooling
        if self.pool_mode == "cls":
            cls = self.cls_token.expand(h.size(0), -1, -1)  # [B, 1, D]
            h = torch.cat([cls, h], dim=1)  # [B, 1+T_total, D]

        h = self.pos_enc(h)          # [B, T_total, temporal_hidden]
        h = self.transformer(h)      # [B, T_total, temporal_hidden]

        # R26g: pooling strategy
        if self.pool_mode == "cls":
            h = self.norm(h[:, 0, :])    # [B, D] CLS token position
        elif self.pool_mode == "mean":
            h = self.norm(h.mean(dim=1))  # [B, D] mean over all positions
        else:  # "last"
            h = self.norm(h[:, -1, :])    # [B, D] last position
        return h


# ---------------------------------------------------------------------------
# 3. Main STEMS Encoder V3
# ---------------------------------------------------------------------------
class STEMSEncoder(nn.Module):
    """
    STEMS v3: Spatial GCN + Per-Node Temporal Transformer encoder.

    The temporal history is embedded in the observation by a
    TemporalHistoryWrapper. The encoder parses the history segment and
    applies a per-building Transformer. This is fully stateless.

    Observation layout:
        [current_obs (198 dims)] [history (T * features_per_step dims)]

    History layout per timestep (features_per_step=11):
        [soc_0, soc_1, ..., soc_4, net_0, net_1, ..., net_4, price]
        i.e., 5 battery SOCs + 5 net consumptions + 1 electricity price

    Per-node temporal input (3 features per timestep per building):
        building i -> [soc_i, net_consumption_i, price]
    """

    def __init__(
        self,
        obs_dim: int,
        node_info: Dict,
        num_buildings: int = 5,
        hidden_dim: int = 64,
        global_hidden: int = 32,
        temporal_window: int = 12,
        temporal_features_per_step: int = 11,
        temporal_hidden: int = 32,
        temporal_heads: int = 4,
        num_gcn_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        output_dim: int = 256,
        history_indices: Optional[List[int]] = None,
        temporal_features_per_node: int = 3,
        per_node_history_map: Optional[List[List[int]]] = None,
        history_ev_mask: Optional[List[List[float]]] = None,
        inject_forecast: bool = False,
        forecast_temporal_dim: int = 72,
        forecast_steps: int = 24,
        forecast_features_per_step: int = 3,
        temporal_num_layers: int = 2,
        temporal_pool_mode: str = "mean",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_buildings = num_buildings
        self.hidden_dim = hidden_dim
        self.temporal_window = temporal_window
        self.temporal_features_per_step = temporal_features_per_step
        self.temporal_hidden = temporal_hidden
        self.temporal_features_per_node = temporal_features_per_node
        self.base_obs_dim = node_info["base_obs_dim"]

        # R26g: forecast injection into temporal transformer
        self.inject_forecast = inject_forecast
        self.forecast_temporal_dim = forecast_temporal_dim  # 72 = price[24]+load[24]+solar[24]
        self.forecast_steps = forecast_steps               # 24
        self.forecast_features_per_step = forecast_features_per_step  # 3

        # Compute where history starts in the observation vector.
        # Layout: [base_obs (70)] [forecast (128)] [history (T * features_per_step)]
        # current_obs_dim = obs_dim - T * features_per_step
        self.history_dim = temporal_window * temporal_features_per_step
        self.current_obs_dim = obs_dim - self.history_dim

        # ---- Index buffers for observation parsing (same as v2) ----
        bld_idx = node_info["building_indices"]  # [N][4]
        self.register_buffer("bld_idx", torch.tensor(bld_idx, dtype=torch.long))

        ev_idx_list = []
        ev_mask_list = []
        for ev in node_info["ev_indices"]:
            if ev is not None:
                ev_idx_list.append(ev)
                ev_mask_list.append(1.0)
            else:
                ev_idx_list.append([0] * 7)
                ev_mask_list.append(0.0)
        self.register_buffer("ev_idx", torch.tensor(ev_idx_list, dtype=torch.long))
        self.register_buffer("ev_mask", torch.tensor(ev_mask_list, dtype=torch.float32))

        self.register_buffer(
            "global_idx", torch.tensor(node_info["global_indices"], dtype=torch.long)
        )

        # ---- Build per-node history index mapping ----
        if per_node_history_map is not None:
            self._build_explicit_history_mapping(per_node_history_map)
        elif history_indices is not None:
            self._build_history_mapping_from_indices(history_indices)
        else:
            self._build_default_history_mapping()

        # ---- EV mask for temporal features (zeros out dummy EV slots) ----
        if history_ev_mask is not None:
            self.register_buffer(
                "hist_ev_mask",
                torch.tensor(history_ev_mask, dtype=torch.float32),  # [N, K]
            )
        else:
            self.hist_ev_mask = None  # no masking needed

        # ---- Temporal Transformer (shared across nodes) ----
        # R26g: max_len must cover backward + forward positions
        max_seq_len = temporal_window + (forecast_steps if inject_forecast else 0)
        self.temporal_transformer = PerNodeTemporalTransformer(
            input_features=temporal_features_per_node,
            temporal_hidden=temporal_hidden,
            temporal_heads=temporal_heads,
            ff_dim=temporal_hidden * 4,
            dropout=dropout,
            max_len=max(max_seq_len + 4, 16),  # +4 for safety margin
            forecast_features=forecast_features_per_step if inject_forecast else 0,
            num_layers=temporal_num_layers,
            pool_mode=temporal_pool_mode,
        )
        self.temporal_proj = nn.Linear(temporal_hidden, hidden_dim)

        # ---- Current observation embedding (same structure as v2) ----
        base_obs_dim = node_info["base_obs_dim"]
        # forecast_dim: features between base_obs and history start
        forecast_dim = self.current_obs_dim - base_obs_dim

        # R26g: when injecting forecast into temporal, subtract the temporal
        # portion (price/load/solar = 72 dims) from global_enc input
        if inject_forecast and forecast_dim >= forecast_temporal_dim:
            global_forecast_dim = forecast_dim - forecast_temporal_dim
        else:
            global_forecast_dim = forecast_dim
        global_raw_dim = len(node_info["global_indices"]) + max(global_forecast_dim, 0)

        self.global_enc = nn.Sequential(
            nn.Linear(global_raw_dim, 64),
            nn.ReLU(),
            nn.Linear(64, global_hidden),
            nn.LayerNorm(global_hidden),
            nn.ReLU(),
        )

        node_raw_dim = 4 + 7 + global_hidden  # building(4) + ev(7) + global_emb
        self.obs_embed = ObservationEmbedding(node_raw_dim, hidden_dim, dropout)

        # ---- Combine projection: concat [node_embed, z_temporal] -> hidden_dim ----
        self.combine_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # ---- Spatial GCN (reused from v2) ----
        self.graph_constructor = AdaptiveGraphConstructor(num_buildings, hidden_dim)
        self.spatial_encoder = SpatialEncoder(hidden_dim, num_gcn_layers, dropout)

        # ---- Gated Fusion: spatial GCN output + temporal output ----
        self.fusion = GatedFusion(hidden_dim)

        # ---- Output projection (no final ReLU) ----
        self.output_proj = nn.Sequential(
            nn.LayerNorm(num_buildings * hidden_dim),
            nn.Linear(num_buildings * hidden_dim, output_dim),
        )
        self.output_dim = output_dim

        # ---- Auxiliary temporal prediction head (R26k) ----
        # Predicts next-hour [price, solar, net_load] from transformer summary.
        # Provides direct supervised gradient to the temporal transformer.
        self._aux_targets = int(os.environ.get("STEMS_AUX_TARGETS", "0"))
        if self._aux_targets > 0:
            self.aux_head = nn.Sequential(
                nn.Linear(temporal_hidden, 32),
                nn.ReLU(),
                nn.Linear(32, self._aux_targets),
            )
            print(f"[STEMSEncoder] Aux prediction head: {self._aux_targets} targets")
        else:
            self.aux_head = None
        # Storage for aux predictions (retrieved by training loop)
        self._last_aux_pred = None

    def _build_default_history_mapping(self):
        """
        Default history layout per timestep (features_per_step=11):
            [soc_0, soc_1, ..., soc_4, net_0, net_1, ..., net_4, price]

        For building i, per-timestep indices are: [i, N+i, 2*N]
        """
        N = self.num_buildings
        K = self.temporal_features_per_node  # default 3
        per_node_indices = []
        for i in range(N):
            per_node_indices.append([i, N + i, 2 * N])  # soc_i, net_i, price
        # [N, K] within a single timestep
        self.register_buffer(
            "history_node_idx",
            torch.tensor(per_node_indices, dtype=torch.long),
        )

    def _build_history_mapping_from_indices(self, history_indices: List[int]):
        """
        Build per-node history mapping from wrapper-provided indices.
        Assumes canonical order: [soc_0..soc_4, net_0..net_4, price].
        """
        N = self.num_buildings
        if len(history_indices) != self.temporal_features_per_step:
            raise ValueError(
                f"history_indices length ({len(history_indices)}) != "
                f"temporal_features_per_step ({self.temporal_features_per_step})"
            )
        per_node_indices = []
        for i in range(N):
            per_node_indices.append([i, N + i, 2 * N])
        self.register_buffer(
            "history_node_idx",
            torch.tensor(per_node_indices, dtype=torch.long),
        )

    def _build_explicit_history_mapping(self, per_node_map: List[List[int]]):
        """
        Build per-node history mapping from explicit per-node index lists.

        per_node_map: [N][K] where K = temporal_features_per_node.
            Each entry is a within-timestep index (0 to features_per_step-1).
        """
        N = self.num_buildings
        K = self.temporal_features_per_node
        F = self.temporal_features_per_step
        if len(per_node_map) != N:
            raise ValueError(
                f"per_node_map has {len(per_node_map)} nodes, expected {N}"
            )
        for i, node_idx in enumerate(per_node_map):
            if len(node_idx) != K:
                raise ValueError(
                    f"per_node_map[{i}] has {len(node_idx)} features, "
                    f"expected {K}"
                )
            max_idx = max(node_idx)
            if max_idx >= F:
                raise ValueError(
                    f"per_node_map[{i}] has index {max_idx} >= "
                    f"features_per_step ({F})"
                )
            if min(node_idx) < 0:
                raise ValueError(
                    f"per_node_map[{i}] has negative index"
                )
        self.register_buffer(
            "history_node_idx",
            torch.tensor(per_node_map, dtype=torch.long),
        )

    def _parse_current_obs(
        self, current_obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Parse the current observation segment.

        Returns:
            node_raw: [B, N, 11] per-building + EV features
            global_raw: [B, G] global + (non-temporal) forecast features
            temporal_forecast: [B, forecast_steps, forecast_features] or None
        """
        B = current_obs.size(0)
        N = self.num_buildings

        # Building features: [B, N, 4]
        bld_feats = current_obs[:, self.bld_idx.view(-1)].view(B, N, 4)

        # EV features: [B, N, 7] (masked for buildings without chargers)
        ev_feats = current_obs[:, self.ev_idx.view(-1)].view(B, N, 7)
        ev_feats = ev_feats * self.ev_mask.view(1, N, 1)

        node_raw = torch.cat([bld_feats, ev_feats], dim=-1)  # [B, N, 11]

        # Global features
        global_shared = current_obs[:, self.global_idx]  # [B, len(global_idx)]
        temporal_forecast = None

        if current_obs.size(1) > self.base_obs_dim:
            forecast_feats = current_obs[:, self.base_obs_dim:]  # [B, forecast_dim]

            if self.inject_forecast and forecast_feats.size(1) >= self.forecast_temporal_dim:
                # R26g: split forecast — first 72 dims (price/load/solar) go to
                # temporal transformer, remaining 56 dims stay in global_enc
                temporal_raw = forecast_feats[:, :self.forecast_temporal_dim]  # [B, 72]
                temporal_forecast = temporal_raw.view(
                    B, self.forecast_steps, self.forecast_features_per_step
                )  # [B, 24, 3]
                global_forecast = forecast_feats[:, self.forecast_temporal_dim:]  # [B, 56]
                global_raw = torch.cat([global_shared, global_forecast], dim=-1)
            else:
                # Original behavior: all forecast features go to global_enc
                global_raw = torch.cat([global_shared, forecast_feats], dim=-1)
        else:
            global_raw = global_shared

        return node_raw, global_raw, temporal_forecast

    def _parse_history(self, history_flat: torch.Tensor) -> torch.Tensor:
        """
        Parse the history segment into per-node temporal features.

        history_flat: [B, T * features_per_step]
        Returns: per_node_history [B, N, T, K]
            K = temporal_features_per_node (default 3, or 10 for rich config).
        """
        B = history_flat.size(0)
        T = self.temporal_window
        F = self.temporal_features_per_step
        N = self.num_buildings
        K = self.temporal_features_per_node

        # Reshape to [B, T, F]
        history = history_flat.view(B, T, F)

        # Extract per-node features using pre-built index mapping
        # history_node_idx: [N, K] (within-step indices for each node)
        idx = self.history_node_idx  # [N, K]

        # Gather all nodes at once
        idx_expanded = idx.unsqueeze(0).unsqueeze(0).expand(B, T, N, K)  # [B, T, N, K]
        history_expanded = history.unsqueeze(2).expand(B, T, N, F)  # [B, T, N, F]
        per_node = torch.gather(history_expanded, dim=3, index=idx_expanded)  # [B, T, N, K]

        # Apply EV mask if present (zeros out dummy EV features for non-EV buildings)
        if self.hist_ev_mask is not None:
            # hist_ev_mask: [N, K] -> broadcast to [1, 1, N, K]
            per_node = per_node * self.hist_ev_mask.unsqueeze(0).unsqueeze(0)

        # Transpose to [B, N, T, K]
        per_node = per_node.permute(0, 2, 1, 3)

        return per_node

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: [B, obs_dim] or [obs_dim]
        Returns: [B, output_dim] feature vector for OmniSafe actor/critic.

        Fully stateless: same computation in rollout and PPO update.
        """
        squeeze = False
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            squeeze = True

        B = obs.size(0)
        N = self.num_buildings

        # ---- Split observation into current and history segments ----
        current_obs = obs[:, :self.current_obs_dim]     # [B, current_obs_dim]
        history_flat = obs[:, self.current_obs_dim:]     # [B, T * features_per_step]

        # ---- 1. Parse current observation ----
        node_raw, global_raw, temporal_forecast = self._parse_current_obs(current_obs)

        # ---- 2. Encode global context and broadcast to nodes ----
        global_emb = self.global_enc(global_raw)  # [B, global_hidden]
        global_broadcast = global_emb.unsqueeze(1).expand(B, N, -1)

        # ---- 3. Current node embedding ----
        node_input = torch.cat([node_raw, global_broadcast], dim=-1)  # [B, N, 43]
        node_embed = self.obs_embed(node_input)  # [B, N, hidden_dim]

        # ---- 4. Per-node temporal transformer ----
        per_node_history = self._parse_history(history_flat)  # [B, N, T, K]

        # Process each node through the SHARED temporal transformer
        # Reshape to process all nodes in one batch: [B*N, T, K]
        K = self.temporal_features_per_node
        temporal_input = per_node_history.reshape(B * N, self.temporal_window, K)

        # R26g: broadcast district-level forecast to all nodes
        forecast_input = None
        if temporal_forecast is not None and self.inject_forecast:
            # temporal_forecast: [B, 24, 3] -> broadcast to [B, N, 24, 3] -> [B*N, 24, 3]
            forecast_input = temporal_forecast.unsqueeze(1).expand(
                B, N, self.forecast_steps, self.forecast_features_per_step
            ).reshape(B * N, self.forecast_steps, self.forecast_features_per_step)

        temporal_summary = self.temporal_transformer(
            temporal_input, forecast_seq=forecast_input
        )  # [B*N, temporal_hidden]
        temporal_summary = temporal_summary.view(B, N, self.temporal_hidden)  # [B, N, temporal_hidden]

        # ---- Auxiliary prediction (R26k): predict next-hour targets from temporal summary ----
        if self.aux_head is not None:
            # Mean-pool across buildings for district-level prediction
            district_temporal = temporal_summary.mean(dim=1)  # [B, temporal_hidden]
            self._last_aux_pred = self.aux_head(district_temporal)  # [B, aux_targets]
        else:
            self._last_aux_pred = None

        # Project temporal summary to hidden_dim
        z_temporal = self.temporal_proj(temporal_summary)  # [B, N, hidden_dim]

        # ---- 5. Combine current embedding with temporal (concatenation) ----
        # NOTE: v3 used residual addition (node_embed + z_temporal) but the
        # network bypassed temporal entirely. Concatenation forces both streams
        # to contribute distinct features.
        node_embed_combined = self.combine_proj(
            torch.cat([node_embed, z_temporal], dim=-1)
        )  # [B, N, hidden_dim]

        # ---- 6. Adaptive graph construction ----
        adj = self.graph_constructor(node_embed_combined)  # [N, N]

        # ---- 7. Spatial GCN ----
        h_spatial_list = []
        for b in range(B):
            h_b = self.spatial_encoder(node_embed_combined[b], adj)  # [N, D]
            h_spatial_list.append(h_b)
        h_spatial = torch.stack(h_spatial_list, dim=0)  # [B, N, D]

        # ---- 8. Gated fusion (spatial GCN + temporal) ----
        r = self.fusion(h_spatial, z_temporal)  # [B, N, hidden_dim]

        # ---- 9. Flatten and project ----
        r_flat = r.reshape(B, N * self.hidden_dim)  # [B, N*D]
        features = self.output_proj(r_flat)  # [B, output_dim]

        if squeeze:
            features = features.squeeze(0)

        return features


# ---------------------------------------------------------------------------
# Smoke Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    def run_smoke_test(label, obs_dim, features_per_step, features_per_node,
                       per_node_map=None, ev_mask=None, inject_forecast=False):
        NUM_BUILDINGS = 5
        TEMPORAL_WINDOW = 12
        CURRENT_OBS_DIM = 198

        mock_node_info = dict(
            building_indices=[[i * 4 + j for j in range(4)] for i in range(5)],
            ev_indices=[
                [20 + i * 7 + j for j in range(7)] if i in [0, 3, 4] else None
                for i in range(5)
            ],
            global_indices=list(range(41, 54)),
            base_obs_dim=70,
        )

        print("=" * 60)
        print(f"STEMSEncoder Smoke Test: {label}")
        print("=" * 60)
        print(f"  features/step: {features_per_step}, features/node: {features_per_node}")
        print(f"  obs_dim: {obs_dim}, inject_forecast: {inject_forecast}")
        print()

        encoder = STEMSEncoder(
            obs_dim=obs_dim,
            node_info=mock_node_info,
            num_buildings=NUM_BUILDINGS,
            hidden_dim=64,
            global_hidden=32,
            temporal_window=TEMPORAL_WINDOW,
            temporal_features_per_step=features_per_step,
            temporal_hidden=32,
            temporal_heads=4,
            num_gcn_layers=3,
            dropout=0.1,
            output_dim=256,
            temporal_features_per_node=features_per_node,
            per_node_history_map=per_node_map,
            history_ev_mask=ev_mask,
            inject_forecast=inject_forecast,
            forecast_temporal_dim=72,
            forecast_steps=24,
            forecast_features_per_step=3,
        )

        # Test 1: Forward pass
        print("[Test 1] Forward pass (batch=4)")
        obs = torch.randn(4, obs_dim)
        out = encoder(obs)
        assert out.shape == (4, 256), f"Expected (4, 256), got {out.shape}"
        print(f"  Input:  {obs.shape} -> Output: {out.shape}  [PASS]")

        # Test 2: Single observation
        print("[Test 2] Single observation (unbatched)")
        out_single = encoder(torch.randn(obs_dim))
        assert out_single.shape == (256,), f"Expected (256,), got {out_single.shape}"
        print(f"  Output: {out_single.shape}  [PASS]")

        # Test 3: Gradient flow
        print("[Test 3] Gradient flow")
        encoder.train()
        out_grad = encoder(torch.randn(4, obs_dim))
        out_grad.sum().backward()
        no_grad = []
        for name, p in encoder.named_parameters():
            if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0):
                no_grad.append(name)
        assert len(no_grad) == 0, f"Dead params: {no_grad}"
        print(f"  All parameters received gradients  [PASS]")

        # Test 4: Parameter count
        total = sum(p.numel() for p in encoder.parameters())
        print(f"[Test 4] Params: {total:,}")

        # Test 5: Determinism
        encoder.eval()
        obs_det = torch.randn(2, obs_dim)
        with torch.no_grad():
            diff = (encoder(obs_det) - encoder(obs_det)).abs().max().item()
        assert diff == 0.0, f"Non-deterministic: {diff}"
        print(f"[Test 5] Determinism  [PASS]")
        print()

    # --- Mode A: Default (3 features/node, backward compat) ---
    run_smoke_test(
        label="Default (3 features/node)",
        obs_dim=198 + 12 * 11,  # 330
        features_per_step=11,
        features_per_node=3,
    )

    # --- Mode B: Rich (10 features/node, with EV mask) ---
    N = 5
    RICH_FPS = 32  # 5*4 + 3*3 + 3
    # Mock per_node_map: [N, 10]
    rich_map = []
    rich_mask = []
    ev_buildings = {0, 3, 4}
    ev_counter = 0
    for i in range(N):
        node_idx = [i, N + i, 2 * N + i, 3 * N + i]  # soc, net, solar, load
        node_mask = [1.0, 1.0, 1.0, 1.0]
        if i in ev_buildings:
            base = 4 * N + ev_counter * 3  # ev features start after building features
            node_idx.extend([base, base + 1, base + 2])
            node_mask.extend([1.0, 1.0, 1.0])
            ev_counter += 1
        else:
            node_idx.extend([0, 0, 0])
            node_mask.extend([0.0, 0.0, 0.0])
        node_idx.extend([RICH_FPS - 3, RICH_FPS - 2, RICH_FPS - 1])  # price, hour_cos, hour_sin
        node_mask.extend([1.0, 1.0, 1.0])
        rich_map.append(node_idx)
        rich_mask.append(node_mask)

    run_smoke_test(
        label="Rich (10 features/node, EV mask)",
        obs_dim=198 + 12 * RICH_FPS,  # 198 + 384 = 582
        features_per_step=RICH_FPS,
        features_per_node=10,
        per_node_map=rich_map,
        ev_mask=rich_mask,
    )

    print("=" * 60)
    print("All tests passed.")
    print("=" * 60)
