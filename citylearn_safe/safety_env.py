# citylearn_safe/safety_env.py
from __future__ import annotations
import os
from typing import Any, Dict, List
import numpy as np
import gymnasium as gym

from .schema_index import soc_indices_from_schema_and_obs_dim


class CityLearnSafetyEnv(gym.Env):
    """Adds a CMDP-style safety cost via info['cost'] (SoC band from observations)."""
    metadata = {"render_modes": []}

    def __init__(
        self,
        base_env: Any,
        *,
        soc_min: float = 0.1,
        soc_max: float = 0.9,
        soc_obs_name: str = "electrical_storage_soc",
    ):
        super().__init__()
        self.base = base_env
        self.soc_min = float(soc_min)
        self.soc_max = float(soc_max)
        self.observation_space = base_env.observation_space
        self.action_space = base_env.action_space

        schema_path = os.environ.get("CITYLEARN_SCHEMA")
        if not schema_path or not os.path.exists(schema_path):
            raise RuntimeError("CITYLEARN_SCHEMA must point to your schema.json.")

        obs_dim = int(self.observation_space.shape[0])
        self._soc_idx, self._per_b_names = soc_indices_from_schema_and_obs_dim(
            schema_path,
            obs_dim,
            soc_name=soc_obs_name,
        )
        if len(self._soc_idx) == 0:
            print(
                f"[CityLearnSafetyEnv] WARNING: No '{soc_obs_name}' indices found "
                f"(per-building active names: {self._per_b_names})."
            )

    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        obs, info = self.base.reset(seed=seed, options=options)
        obs = np.asarray(obs, dtype=np.float32)

        soc_vals = self._soc_values_from_obs(obs)
        metrics = self._soc_metrics(soc_vals)
        cost = self._soc_band_cost(metrics)

        info = dict(info)
        info["metrics"] = metrics
        info["cost"] = float(cost)
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = self.base.step(action)
        obs = np.asarray(obs, dtype=np.float32)

        soc_vals = self._soc_values_from_obs(obs)
        metrics = self._soc_metrics(soc_vals)
        cost = self._soc_band_cost(metrics)

        info = dict(info)
        info["metrics"] = metrics
        info["cost"] = float(cost)
        return obs, float(r), bool(term), bool(trunc), info

    # ---- SoC helpers
    def _soc_values_from_obs(self, obs: np.ndarray) -> List[float]:
        vals: List[float] = []
        for idx in self._soc_idx:
            if 0 <= idx < obs.shape[0]:
                vals.append(float(obs[idx]))
        return vals

    def _soc_metrics(self, vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {
                "soc_mean": 0.5,
                "soc_min_obs": 0.5,
                "soc_max_obs": 0.5,
                "num_storages": 0.0,
            }
        return {
            "soc_mean": float(np.mean(vals)),
            "soc_min_obs": float(np.min(vals)),
            "soc_max_obs": float(np.max(vals)),
            "num_storages": float(len(vals)),
        }

    def _soc_band_cost(self, stats: Dict[str, float]) -> float:
        if stats.get("num_storages", 0.0) <= 0.0:
            return 0.0
        low_violation = max(0.0, self.soc_min - stats["soc_min_obs"])
        high_violation = max(0.0, stats["soc_max_obs"] - self.soc_max)
        band = max(1e-6, (self.soc_max - self.soc_min))
        return (low_violation + high_violation) / band
