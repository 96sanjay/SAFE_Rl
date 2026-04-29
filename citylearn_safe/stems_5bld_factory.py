"""Reusable 5-building STEMS v3 construction helpers.

This module lifts the 5-building ObsIndex/STEMS v3 wiring out of ad-hoc
training scripts so new training paths can construct the exact same encoder
contract without importing from ``scripts/`` at runtime.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass

import citylearn_safe.schema_index as schema_index_module
from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
from citylearn_safe.safety_env import CityLearnSafetyEnv
from citylearn_safe.schema_index import build_index
from citylearn_safe.stems_encoder_5bld import build_node_indices
from citylearn_safe.stems_encoder import STEMSEncoder
from scripts.make_env import make_base_env


@dataclass(frozen=True)
class Stems5BldSpec:
    """Resolved 5-building STEMS observation/graph specification."""

    obs_index: object
    node_info: dict
    obs_dim: int
    act_dim: int
    num_buildings: int
    num_evs: int
    temporal_window: int
    history_indices: list[int]
    encoder_obs_dim: int


@contextmanager
def _temporary_env(key: str, value: str):
    old_value = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_value


def build_obs_index_5bld_spec(schema_path: str, temporal_window: int = 12) -> Stems5BldSpec:
    """Build the exact 5-building observation contract used by STEMS v3.

    The resulting encoder observation contract matches the existing repo path:
    base env -> safety env -> forecast obs -> temporal history tail.
    """
    with _temporary_env("CITYLEARN_SCHEMA", schema_path):
        base_env = make_base_env(central_agent=True)
        safety_env = CityLearnSafetyEnv(base_env)

        city = base_env
        for _ in range(20):
            if hasattr(city, "buildings") and len(getattr(city, "buildings", [])) > 0:
                break
            city = getattr(city, "env", getattr(city, "base", getattr(city, "unwrapped", None)))
            if city is None:
                break
        num_buildings = len(city.buildings) if city and hasattr(city, "buildings") else 0
        if num_buildings != 5:
            raise ValueError(
                f"STEMS 5-building path requires a 5-building schema, got {num_buildings} buildings "
                f"from {schema_path}."
            )

        schema_index_module._CACHE = None
        obs_index = build_index(safety_env, expected_buildings=num_buildings)
        num_evs = len(obs_index.ev)

        forecast_env = ForecastObsWrapper(safety_env, forecast_horizon=24)
        obs_space = forecast_env.observation_space
        obs_dim = int(obs_space[0].shape[0]) if isinstance(obs_space, (list, tuple)) else int(obs_space.shape[0])

        act_space = forecast_env.action_space
        act_dim = int(act_space[0].shape[0]) if isinstance(act_space, (list, tuple)) else int(act_space.shape[0])
        if act_dim != 9:
            raise ValueError(f"Expected 5-building act_dim=9, got {act_dim} for {schema_path}.")

        history_indices: list[int] = []
        for i in range(num_buildings):
            history_indices.append(obs_index.electrical_storage_soc[i])
            history_indices.append(obs_index.net_electricity_consumption[i])
        history_indices.append(obs_index.electricity_pricing)
        if len(history_indices) != 11:
            raise ValueError(
                f"Expected 11 temporal history features for 5-building STEMS, got {len(history_indices)}."
            )

        encoder_obs_dim = obs_dim + len(history_indices) * temporal_window
        node_info = build_node_indices(obs_index, num_buildings)

        return Stems5BldSpec(
            obs_index=obs_index,
            node_info=node_info,
            obs_dim=obs_dim,
            act_dim=act_dim,
            num_buildings=num_buildings,
            num_evs=num_evs,
            temporal_window=temporal_window,
            history_indices=history_indices,
            encoder_obs_dim=encoder_obs_dim,
        )


def build_stems_encoder_5bld(
    spec: Stems5BldSpec,
    hidden_dim: int = 64,
    global_hidden: int = 32,
    temporal_hidden: int = 32,
    temporal_heads: int = 4,
    num_gcn_layers: int = 3,
    output_dim: int = 256,
    dropout: float = 0.1,
) -> STEMSEncoder:
    """Instantiate the exact 5-building STEMS v3 encoder."""
    if spec.num_buildings != 5:
        raise ValueError(f"Expected 5-building STEMS spec, got {spec.num_buildings} buildings.")
    if spec.act_dim != 9:
        raise ValueError(f"Expected act_dim=9 for 5-building STEMS spec, got {spec.act_dim}.")
    if len(spec.history_indices) != 11:
        raise ValueError(
            f"Expected 11 temporal history features for 5-building STEMS spec, got {len(spec.history_indices)}."
        )

    return STEMSEncoder(
        obs_dim=spec.encoder_obs_dim,
        node_info=spec.node_info,
        num_buildings=spec.num_buildings,
        hidden_dim=hidden_dim,
        global_hidden=global_hidden,
        temporal_window=spec.temporal_window,
        temporal_features_per_step=len(spec.history_indices),
        temporal_hidden=temporal_hidden,
        temporal_heads=temporal_heads,
        num_gcn_layers=num_gcn_layers,
        dropout=dropout,
        output_dim=output_dim,
        history_indices=spec.history_indices,
    )
