#!/usr/bin/env python3
"""
Standardized V2G benchmark evaluator (thesis-grade).

Produces a defensible benchmark table for the R28 algorithm suite by running each
trained policy through:

  - 3 evaluation windows (post-hoc sliced from a single full-year rollout).
    The schema starts on 2024-08-01 (hour 0):
        full_year   = steps [0, 8760)
        period_a    = steps [0, 2160)        (Aug 1 - Oct 30 2024)
        period_b    = steps [4344, 6504)     (Jan 29 - Apr 29 2025)

  - Oracle two-rollout per algo (full year):
        policy rollout  - the model's deterministic mean-action policy
        oracle rollout  - same policy, but EV charger actions forced to +1.0
                          whenever a charger has a connected EV
        from these we compute:
            oracle_deficit_kwh        = simulator-unavoidable kWh
            avoidable_wrt_oracle_kwh  = max(0, policy_def_kwh - oracle_def_kwh)
            avoidable_wrt_oracle_pct  = 100 * avoidable / policy_def_kwh

  - Env-var isolation:
        os.environ snapshot/restore around each model so per-model `extra_env`
        cannot leak into the next model in the same Python process.

  - Per-departure C0 fix:
        FIXED rate uses info["ev_departure_violation_count_deficit"]
        (per-EV-departure counter). The old buggy step-based rate is reported
        side-by-side as `c0_per_dep_rate_OLD_BUGGY`.

Outputs (under scripts/eval_outputs/, one timestamp per run):
    eval_r28_benchmark_<TS>.log     full stdout dump
    eval_r28_benchmark_<TS>.json    nested dict {algo: {window: metrics}}
    eval_r28_benchmark_<TS>.csv     one row per (algo, window)

Constraints honoured:
    - Does NOT modify scripts/eval_r28b_vs_sac.py or scripts/eval_r28_fixed.py
    - Does NOT modify any env code
    - Does NOT include the 3 baselines (RBC / charge-on-arrival / zero-action);
      a clean integration point is left at `actor_type == "baseline_ext"` so
      the baselines agent can plug in via the same `MODELS` dict.

Usage:
    python scripts/eval_r28_benchmark.py                # all 10 models
    python scripts/eval_r28_benchmark.py --skip-csac    # skip CSAC-LB if not yet trained
    python scripts/eval_r28_benchmark.py --only "R28 PPO (vanilla)"   # smoke-test
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COST_KEYS = [
    "cost_ev_departure",          # C0
    "cost_ev_dense",              # C1
    "cost_stems_battery",         # C2
    "cost_stems_building_power",  # C3
    "cost_stems_grid_power",      # C4
]
COST_LABELS = ["C0", "C1", "C2", "C3", "C4"]
COST_LIMITS = [100, 999999, 3500, 18000, 13000]

# Post-hoc evaluation windows (start_inclusive, end_exclusive) over a full
# 8760-step rollout. Single deterministic rollout, sliced after the fact.
#
# Calendar mapping: the CityLearn 5-building schema starts Aug 1 2024 (hour 0).
#   step 0     = 2024-08-01 00:00
#   step 2160  = 2024-10-30 (90 days later) -> end of period_a
#   step 4344  = 2025-01-29 (~181 days later) -> start of period_b
#   step 6504  = 2025-04-29 (~271 days later) -> end of period_b
#   step 8760  = 2025-07-31 (full year)
#
# period_a  : Aug 1 – Oct 30 2024 (autumn/early-cool-season window)
# period_b  : late-Jan – late-Apr 2025 (winter/early-spring window)
WINDOWS: Dict[str, Tuple[int, int]] = {
    "full_year": (0, 8760),
    "period_a": (0, 2160),        # Aug 1 - Oct 30 2024 (was "winter_q1")
    "period_b": (4344, 6504),     # Jan 29 - Apr 29 2025 (was "summer_q3")
}

# Shared env vars. Lifted verbatim from scripts/eval_r28b_vs_sac.py (lines 38-91)
# so this script reproduces the canonical evaluation environment.
COMMON_ENV: Dict[str, str] = {
    "CITYLEARN_CENTRAL_AGENT": "1",
    "CITYLEARN_REWARD_TYPE": "stems",
    "CITYLEARN_EXPORT_FACTOR": "0.7",
    "CITYLEARN_TEMPORAL_WINDOW": "0",
    "CITYLEARN_STEMS_P_BUILDING_MAX": "4.6083",
    "CITYLEARN_STEMS_P_GRID_MAX": "10.2352",
    "CITYLEARN_STEMS_SOC_LOW": "0.0",
    "CITYLEARN_STEMS_SOC_HIGH": "0.95",
    "CITYLEARN_STEMS_PNORM_P": "4.0",
    "CITYLEARN_PID_LAGRANGE": "1",
    "CITYLEARN_EV_SAUTE": "0",
    # Reward weights (R28b 9-term)
    "STEMS_LAMBDA_EV": "2.0",
    "STEMS_ALPHA_EV_SMART": "1.5",
    "STEMS_EV_SLACK_ARB_SCALE": "1.0",
    "STEMS_ALPHA_V2G_CONTEXT": "1.5",
    "STEMS_ALPHA_BARRIER": "0.5",
    "STEMS_ALPHA_GRID": "0.5",
    "STEMS_SG_THRESHOLD": "0.5",
    "STEMS_SG_EXPORT_CREDIT": "0.5",
    "STEMS_ALPHA_BUILD": "0.3",
    "STEMS_SB_ASYMMETRIC": "1",
    "STEMS_BETA_RAMP": "0.3",
    "STEMS_XI_RENEWABLE": "0.2",
    # Disabled
    "STEMS_ALPHA_EV_GUARD": "0.0",
    "STEMS_ALPHA_GRID_MILD": "0.0",
    "STEMS_MU_ECONOMIC": "0.0",
    "STEMS_ALPHA_PEAK_SHAVE": "0.0",
    "STEMS_ALPHA_LOAD_SHIFT": "0.0",
    # Cost weights
    "COST_W_C2": "0.0",
    "COST_W_C3": "5.0",
    "CITYLEARN_W_COST_EV": "1.0",
    "CITYLEARN_W_COST_SOC": "10.0",
    "CITYLEARN_W_COST_BUILDING": "0.5",
    "CITYLEARN_W_COST_GRID": "0.05",
    "CITYLEARN_EV_COST_SCALE": "3.0",
    "CITYLEARN_EV_MISSING_ACTION_MODE": "assume_full",
    "CITYLEARN_INCLUDE_EV_COST": "1",
    # Controllability
    "CITYLEARN_C3_CONTROLLABLE": "1",
    "CITYLEARN_BATT_CLAMP": "0",
    # Observation/Action
    "CITYLEARN_SPATIAL_OBS": "0",
    "CITYLEARN_WM_DISABLE": "1",
    "CITYLEARN_EV_ACTION_CLAMP": "0",
    "CITYLEARN_ACTION_MASK": "0",
    "CITYLEARN_POLICY_ACTION_MASK": "0",
    # Suppress noisy logs
    "CITYLEARN_KPI_FLUSH_EVERY_STEP": "0",
    "CITYLEARN_DEBUG_ACTION_CLIP": "0",
}

# 10 algos. Keys + extra_env mirror scripts/eval_r28b_vs_sac.py:MODELS.
# CSAC-LB checkpoint path is the R28-reward variant (see TASK).
MODELS: Dict[str, Dict[str, Any]] = {
    "PPO_vanilla": {
        "ckpt": "runs/r28_ppo/5bld/PPO-{CityLearnSafety-V2G-v2}/seed-000-2026-04-17-04-42-44/torch_save/epoch-100.pt",
        "actor_type": "ppo",
        "extra_env": {
            "CITYLEARN_PID_LAGRANGE": "0",
            "COST_W_C2": "5.0",
            "COST_W_C1_DENSE": "0.0",
        },
        "training_seed": 0,
    },
    "PPOLag": {
        "ckpt": "runs/r28_ppolag/5bld/PPOLag-{CityLearnSafety-V2G-v2}/seed-000-2026-04-16-04-54-39/torch_save/epoch-100.pt",
        "actor_type": "ppo",
        "extra_env": {
            "CITYLEARN_PID_LAGRANGE": "0",
            "COST_W_C2": "5.0",
            "COST_W_C1_DENSE": "0.0",
        },
        "training_seed": 0,
    },
    "TRPOLag": {
        "ckpt": "runs/r28_trpolag/5bld/TRPOLag-{CityLearnSafety-V2G-v2}/seed-000-2026-04-17-04-42-41/torch_save/epoch-100.pt",
        "actor_type": "ppo",
        "extra_env": {
            "CITYLEARN_PID_LAGRANGE": "0",
            "COST_W_C2": "5.0",
            "COST_W_C1_DENSE": "0.0",
        },
        "training_seed": 0,
    },
    "CPPOPID": {
        "ckpt": "runs/r28_cppopid/5bld/CPPOPID-{CityLearnSafety-V2G-v2}/seed-000-2026-04-16-04-54-40/torch_save/epoch-100.pt",
        "actor_type": "ppo",
        "extra_env": {
            "CITYLEARN_PID_LAGRANGE": "0",
            "COST_W_C2": "5.0",
            "COST_W_C1_DENSE": "0.0",
        },
        "training_seed": 0,
    },
    "CPO": {
        "ckpt": "runs/r28_cpo/5bld/CPO-{CityLearnSafety-V2G-v2}/seed-000-2026-04-15-23-41-52/torch_save/epoch-100.pt",
        "actor_type": "ppo",
        "extra_env": {},
        "training_seed": 0,
    },
    "PPOLagMulti_R28b": {
        "ckpt": "runs/r28b/5bld/PPOLagMulti-{CityLearnSafety-V2G-v2}/seed-042-2026-04-15-09-32-30/torch_save/epoch-100.pt",
        "actor_type": "ppo",
        "extra_env": {},
        "training_seed": 42,
    },
    "SACLagMulti": {
        # Note: only epoch-95 exists for this run, per srv07 sync
        "ckpt": "runs/r28c_sac/5bld/SACLagMulti-{CityLearnSafety-V2G-v2}/seed-042-2026-04-15-10-44-06/torch_save/epoch-95.pt",
        "actor_type": "sac",
        "extra_env": {"STEMS_ALPHA_LOAD_SHIFT": "0.5"},
        "training_seed": 42,
    },
    "SACLag": {
        "ckpt": "runs/r28_saclag/5bld/SACLag-{CityLearnSafety-V2G-v2}/seed-000-2026-04-17-04-42-43/torch_save/epoch-100.pt",
        "actor_type": "sac",
        "extra_env": {
            "CITYLEARN_PID_LAGRANGE": "0",
            "COST_W_C2": "5.0",
            "COST_W_C1_DENSE": "0.0",
        },
        "training_seed": 0,
    },
    "PPOSaute": {
        "ckpt": "runs/r28_pposaute/5bld/PPOSaute-{CityLearnSafety-V2G-v2}/seed-000-2026-04-17-04-15-16/torch_save/epoch-100.pt",
        "actor_type": "ppo",
        "extra_env": {
            "CITYLEARN_PID_LAGRANGE": "0",
            "COST_W_C2": "5.0",
            "COST_W_C1_DENSE": "0.0",
        },
        "training_seed": 0,
        # FOOTNOTE: PPOSaute eval uses saute_pad=True (constant safety_state=1.0)
        # appended to obs to match the actor's input_dim=obs_dim+1. This is a
        # biased approximation - ideal eval would track safety_state dynamically.
        "footnote": "saute_pad: constant safety_state=1.0; biased proxy",
    },
    "CSAC_LB_R28parity": {
        # User-trained 100-epoch CSAC-LB on EXACT R28 9-term reward
        # (verified config.json env_overrides match COMMON_ENV exactly).
        "ckpt": "runs/csac_lb_v2g/CSACLBV2G-{CityLearnSafety-V2G-v2}/seed-042-2026-04-18-11-01-34/torch_save/epoch-100.pt",
        "actor_type": "sac",
        "extra_env": {
            # Match config.json env_overrides exactly (R28 9-term reward + COST_W_*)
            "CITYLEARN_PID_LAGRANGE": "0",
            "COST_W_C2": "5.0",
            "COST_W_C3": "5.0",
            "COST_W_C1_DENSE": "0.0",
        },
        "training_seed": 42,
    },
    "CSAC_LB_R28reward": {
        # Currently training; checkpoint expected at the path below.
        # If missing, --skip-csac (or auto-skip-on-missing) will exclude it.
        "ckpt_glob": "runs/csac_lb_v2g_r28reward/CSACLBV2G-{CityLearnSafety-V2G-v2}/seed-042-*/torch_save/epoch-100.pt",
        "actor_type": "sac",
        "extra_env": {
            "CITYLEARN_PID_LAGRANGE": "0",
            "STEMS_MU_ECONOMIC": "1.5",
            "STEMS_BETA_RAMP": "0.3",
            "STEMS_XI_RENEWABLE": "0.3",
            "STEMS_LAMBDA_EV": "2.0",
            "STEMS_ALPHA_EV_GUARD": "0.5",
            "STEMS_ALPHA_EV_SMART": "1.5",
            "COST_W_C0": "3.0",
            "COST_W_C2": "1.0",
            "COST_W_C3": "1.0",
            "COST_W_C4": "1.0",
            "COST_W_C1_DENSE": "0.0",
        },
        "training_seed": 42,
    },
}

# Baselines (rule-based) — populated lazily so the script does not require
# baselines_v2g.py at import time.
def _load_baselines():
    try:
        from scripts.baselines_v2g import BASELINES as _BASELINES
        for bname, bcls in _BASELINES.items():
            MODELS[f"BL_{bname}"] = {
                "actor_type": "baseline_ext",
                "baseline_factory": bcls,
                "extra_env": {},
                "training_seed": None,
            }
    except ImportError as _e:
        print(f"[warn] baselines_v2g.py not importable: {_e}; baselines skipped")

# TODO: Best-checkpoint selection.
#   For now this script uses epoch-100 across the board (epoch-95 for
#   SACLagMulti since that is what exists). A future improvement is to scan
#   torch_save/*.pt and pick the epoch with the best (EpRet, EpCost) trade-off
#   on a held-out validation rollout.


# ---------------------------------------------------------------------------
# Actor classes (copied from eval_r28b_vs_sac.py to keep this script
# self-contained; do NOT import from there since it is historical)
# ---------------------------------------------------------------------------

class PPOActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        layers: List[nn.Module] = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Deterministic mean action: tanh(mean.net(obs))
        return torch.tanh(self.net(x))


class SACActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        self.act_dim = act_dim
        layers: List[nn.Module] = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, act_dim * 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Deterministic mean action: tanh(out[..., :act_dim])
        out = self.net(x)
        mean = out[..., :self.act_dim]
        return torch.tanh(mean)


# ---------------------------------------------------------------------------
# Env / model helpers
# ---------------------------------------------------------------------------

def normalize_obs(obs_np: np.ndarray, obs_norm: Any, device: torch.device) -> torch.Tensor:
    obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)
    if obs_norm is None:
        return obs_t.unsqueeze(0)
    mean = obs_norm["_mean"].to(device).float()
    std = obs_norm["_std"].to(device).float()
    clip_val = obs_norm.get("_clip", torch.tensor(10.0)).to(device).float()
    obs_len = obs_t.shape[0]
    norm_len = mean.shape[0]
    if obs_len > norm_len:
        normed = torch.clamp((obs_t[:norm_len] - mean) / (std + 1e-8), -clip_val, clip_val)
        obs_t = torch.cat([normed, obs_t[norm_len:]], dim=0)
    elif obs_len < norm_len:
        clp = clip_val[:obs_len] if clip_val.dim() > 0 else clip_val
        obs_t = torch.clamp((obs_t - mean[:obs_len]) / (std[:obs_len] + 1e-8), -clp, clp)
    else:
        obs_t = torch.clamp((obs_t - mean) / (std + 1e-8), -clip_val, clip_val)
    return obs_t.unsqueeze(0)


def get_citylearn_env(env: Any) -> Any:
    cur = env
    seen = set()
    for _ in range(40):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        if hasattr(cur, "buildings") and hasattr(cur, "evaluate"):
            return cur
        for attr in ("base", "env", "unwrapped", "_env"):
            nxt = getattr(cur, attr, None)
            if nxt is not None and nxt is not cur:
                cur = nxt
                break
        else:
            break
    return None


def build_eval_env() -> Any:
    """Build the standard evaluation env. Reads CITYLEARN_* env vars."""
    import citylearn_safe.omni_env  # noqa: F401  (registers env)
    import citylearn_safe.cmdp_env  # noqa: F401
    from scripts.make_env import make_base_env
    from citylearn_safe.safety_env import CityLearnSafetyEnv
    from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper

    base_env = make_base_env(central_agent=True)
    safety_env = CityLearnSafetyEnv(base_env)
    env = ForecastObsWrapper(safety_env, forecast_horizon=24)
    return env


def find_ev_action_indices(env: Any) -> List[int]:
    """Walk the wrapper chain to find the safety env's EV charger indices."""
    cur = env
    for _ in range(20):
        if cur is None:
            break
        if hasattr(cur, "_ev_charger_action_indices"):
            return list(getattr(cur, "_ev_charger_action_indices") or [])
        cur = getattr(cur, "env", None)
    return []


def load_actor_from_ckpt(
    ckpt_path: str,
    actor_type: str,
    obs_dim: int,
    act_dim: int,
    device: torch.device,
) -> Tuple[nn.Module, Any, bool]:
    """Returns (model, obs_normalizer, saute_pad_flag)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    pi_sd = ckpt["pi"]

    # Detect Saute actor: ckpt first layer expects obs_dim+1 (safety state).
    saute_pad = False
    for k, v in pi_sd.items():
        if (k == "mean.0.weight" or k == "net.0.weight") and v.shape[1] == obs_dim + 1:
            saute_pad = True
            break
    actor_input_dim = obs_dim + (1 if saute_pad else 0)

    if actor_type == "ppo":
        model = PPOActor(actor_input_dim, act_dim)
        net_sd: Dict[str, torch.Tensor] = {}
        for k, v in pi_sd.items():
            if k.startswith("mean."):
                net_sd["net." + k[5:]] = v
            elif k.startswith("net."):
                net_sd[k] = v
        if not net_sd:
            net_sd = {k: v for k, v in pi_sd.items() if "weight" in k or "bias" in k}
    elif actor_type == "sac":
        model = SACActor(actor_input_dim, act_dim)
        net_sd = {k: v for k, v in pi_sd.items() if k.startswith("net.")}
    else:
        raise ValueError(f"Unknown actor_type={actor_type!r}")

    missing, unexpected = model.load_state_dict(net_sd, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys when loading actor: {missing}")
    if unexpected:
        print(f"  [warn] unexpected ckpt keys ignored: {unexpected}")
    obs_norm = ckpt.get("obs_normalizer", None)
    return model.to(device).eval(), obs_norm, saute_pad


def policy_action(
    model: nn.Module,
    obs_np: np.ndarray,
    obs_norm: Any,
    saute_pad: bool,
    device: torch.device,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray:
    obs_normed = normalize_obs(obs_np, obs_norm, device)
    if saute_pad:
        pad = torch.ones(obs_normed.shape[0], 1, device=device, dtype=obs_normed.dtype)
        obs_normed = torch.cat([obs_normed, pad], dim=-1)
    with torch.no_grad():
        action = model(obs_normed)
    action_np = action.squeeze(0).cpu().numpy()
    return np.clip(action_np, action_low, action_high)


# ---------------------------------------------------------------------------
# Env-var isolation
# ---------------------------------------------------------------------------

@contextmanager
def env_snapshot():
    """Snapshot os.environ on enter; restore on exit. No leakage between models."""
    saved = os.environ.copy()
    try:
        yield
    finally:
        # Remove keys added inside the block
        for k in list(os.environ.keys()):
            if k not in saved:
                del os.environ[k]
        # Restore changed values
        for k, v in saved.items():
            if os.environ.get(k) != v:
                os.environ[k] = v


def env_vars_hash(applied: Dict[str, str]) -> str:
    """Stable short hash of the applied env-var set for the manifest."""
    blob = json.dumps(applied, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Rollout (single full year, captures per-step info for post-hoc slicing)
# ---------------------------------------------------------------------------

def run_full_year_rollout(
    env: Any,
    model: nn.Module,
    obs_norm: Any,
    saute_pad: bool,
    device: torch.device,
    *,
    use_oracle: bool = False,
    ev_action_indices: List[int] | None = None,
    seed: int = 0,
    max_steps: int = 8760,
) -> Dict[str, Any]:
    """One full-year deterministic rollout.

    If use_oracle: after the policy emits its action, override action[ev_idx]=+1.0
    for each charger that currently has an EV connected (per safety_env sim
    state).
    """
    from citylearn_safe.extractors import (
        unwrap_to_raw_citylearn_env,
        _oracle_connected_now,
        _build_ev_action_index_by_charger_id,
        _normalize_action_names,
    )

    # Reset (the env supports `seed=` kwarg via Gym API)
    try:
        obs, info = env.reset(seed=seed)
    except TypeError:
        obs, info = env.reset()

    action_low = env.action_space.low
    action_high = env.action_space.high

    # For oracle: charger_id -> action index
    raw_env = unwrap_to_raw_citylearn_env(env)
    oracle_idx_by_charger: Dict[str, int] = {}
    if use_oracle:
        action_names = _normalize_action_names(getattr(raw_env, "action_names", []))
        oracle_idx_by_charger = _build_ev_action_index_by_charger_id(action_names)

    # Identify per-building / per-charger structure for trajectory logging.
    cl_env_for_meta = get_citylearn_env(env)
    n_buildings = len(cl_env_for_meta.buildings) if cl_env_for_meta is not None else 0
    # Flat list of (building_idx, charger_idx) pairs in stable order so that
    # ev_soc[:, k] always refers to the same physical charger across runs.
    charger_index: List[Tuple[int, int]] = []
    if cl_env_for_meta is not None:
        for bi, b in enumerate(cl_env_for_meta.buildings):
            chs = getattr(b, "electric_vehicle_chargers", None) or []
            for ci, _ in enumerate(chs):
                charger_index.append((bi, ci))
    n_chargers = len(charger_index)

    # Per-step buffers
    rewards: List[float] = []
    actions: List[np.ndarray] = []
    info_records: List[Dict[str, float]] = []
    # Per-step trajectory buffers — fixed-shape numpy arrays for compact .npy
    # serialisation downstream. NaN-pad missing values so shape stays uniform.
    net_load_traj = np.full((max_steps, max(n_buildings, 1)), np.nan, dtype=np.float32)
    ev_soc_traj = np.full((max_steps, max(n_chargers, 1)), np.nan, dtype=np.float32)
    cl_kpis: Dict[str, float] = {}

    for step in range(max_steps):
        action_np = policy_action(
            model, obs, obs_norm, saute_pad, device, action_low, action_high
        )

        if use_oracle and oracle_idx_by_charger:
            for cid, a_idx in oracle_idx_by_charger.items():
                if 0 <= a_idx < len(action_np) and _oracle_connected_now(raw_env, cid):
                    action_np[a_idx] = 1.0

        obs, reward, terminated, truncated, info = env.step(action_np)
        rewards.append(float(reward))
        actions.append(action_np.copy())

        # Per-step per-building net consumption (NaN if attribute missing).
        # CityLearn appends one entry per simulator step, so the value for
        # the step we just executed lives at index time_step-1.
        if cl_env_for_meta is not None:
            ts = getattr(cl_env_for_meta, "time_step", step + 1)
            for bi in range(n_buildings):
                try:
                    nec = cl_env_for_meta.buildings[bi].net_electricity_consumption
                    if nec is not None and len(nec) > 0:
                        idx = min(ts - 1, len(nec) - 1)
                        if idx >= 0:
                            net_load_traj[step, bi] = float(nec[idx])
                except (AttributeError, IndexError, TypeError):
                    pass  # leave as NaN

            for k, (bi, ci) in enumerate(charger_index):
                try:
                    ch = cl_env_for_meta.buildings[bi].electric_vehicle_chargers[ci]
                    ev = getattr(ch, "connected_electric_vehicle", None)
                    if ev is not None:
                        soc_arr = ev.battery.soc
                        if soc_arr is not None and len(soc_arr) > 0:
                            idx = min(ts - 1, len(soc_arr) - 1)
                            if idx >= 0:
                                ev_soc_traj[step, k] = float(soc_arr[idx])
                except (AttributeError, IndexError, TypeError):
                    pass  # leave as NaN (no EV connected at this step)

        # Record exactly the keys we need (slim, json-able)
        rec = {
            "ev_departure_departures": int(info.get("ev_departure_departures", 0)),
            "ev_departure_violation_count_deficit": int(
                info.get("ev_departure_violation_count_deficit", 0)
            ),
            "building_power_violation_count": float(
                info.get("building_power_violation_count", 0.0)
            ),
            "ev_departure_deficit_kwh": float(info.get("ev_departure_deficit_kwh", 0.0)),
            "ev_avoidable_deficit_kwh": float(info.get("ev_avoidable_deficit_kwh", 0.0)),
            "ev_unavoidable_deficit_kwh": float(info.get("ev_unavoidable_deficit_kwh", 0.0)),
            "cost_ev_departure_avoidable": float(
                info.get("cost_ev_departure_avoidable", 0.0)
            ),
            "cost_ev_departure_unavoidable": float(
                info.get("cost_ev_departure_unavoidable", 0.0)
            ),
        }
        for k in COST_KEYS:
            rec[k] = float(info.get(k, 0.0))
        info_records.append(rec)

        if terminated or truncated:
            break

    # Truncate trajectories to actual rollout length.
    actual_len = len(rewards)
    net_load_traj = net_load_traj[:actual_len]
    ev_soc_traj = ev_soc_traj[:actual_len]

    # CityLearn KPIs (only meaningful for the FULL year, since they are
    # episode-level evaluations against the whole simulation period).
    # We now keep BOTH district- and building-level rows so behavioural
    # figures can compare per-building outcomes.
    cl_env = get_citylearn_env(env)
    if cl_env is not None and hasattr(cl_env, "evaluate"):
        try:
            kpi_df = cl_env.evaluate()
            if kpi_df is not None:
                for _, row in kpi_df.iterrows():
                    level = str(row.get("level", ""))
                    cf = str(row.get("cost_function", "unknown"))
                    bname = str(row.get("name", "unknown"))
                    v = row.get("value", None)
                    if v is None or (isinstance(v, float) and not np.isfinite(v)):
                        continue
                    if level == "district":
                        cl_kpis[f"District|{cf}"] = float(v)
                    else:
                        cl_kpis[f"{bname}|{cf}"] = float(v)
        except Exception as e:
            print(f"  [warn] KPI eval failed: {e}")

    return {
        "rewards": rewards,
        "actions": actions,
        "info_records": info_records,
        "cl_kpis": cl_kpis,
        "ep_len": actual_len,
        "net_load_traj": net_load_traj,         # shape (T, n_buildings)
        "ev_soc_traj": ev_soc_traj,             # shape (T, n_chargers); NaN if disconnected
        "charger_index": charger_index,         # [(building_idx, charger_idx), ...]
        "n_buildings": n_buildings,
    }


# ---------------------------------------------------------------------------
# Metrics: per (algo, window)
# ---------------------------------------------------------------------------

def compute_window_metrics(
    policy_roll: Dict[str, Any],
    oracle_roll: Dict[str, Any],
    window: Tuple[int, int],
    ev_cost_scale: float,
    cl_kpis_full_year: Dict[str, float],
    is_full_year: bool,
) -> Dict[str, Any]:
    s, e = window
    # Clamp window to actual rollout length
    e_pol = min(e, policy_roll["ep_len"])
    e_orc = min(e, oracle_roll["ep_len"])

    pol_recs = policy_roll["info_records"][s:e_pol]
    orc_recs = oracle_roll["info_records"][s:e_orc]
    pol_rew = policy_roll["rewards"][s:e_pol]
    pol_acts = policy_roll["actions"][s:e_pol]

    n = len(pol_recs)
    if n == 0:
        return {"error": f"empty window {window}"}

    # Per-cost totals + per-step violation %
    out: Dict[str, Any] = {
        "window_start": s,
        "window_end": e_pol,
        "window_steps": n,
        "ep_ret": float(sum(pol_rew)),
    }
    for i, key in enumerate(COST_KEYS):
        costs = np.array([r[key] for r in pol_recs], dtype=float)
        out[f"cost_{COST_LABELS[i]}_total"] = float(costs.sum())
        out[f"viol_pct_{COST_LABELS[i]}"] = float(100.0 * np.mean(costs > 0.0))
    out["ep_cost_total"] = float(
        sum(out[f"cost_{lbl}_total"] for lbl in COST_LABELS)
    )

    # ----- Joint constraint satisfaction (all 4 active channels) -----
    c0_arr = np.array([r["cost_ev_departure"] for r in pol_recs], dtype=float)
    c2_arr = np.array([r["cost_stems_battery"] for r in pol_recs], dtype=float)
    c3_arr = np.array([r["cost_stems_building_power"] for r in pol_recs], dtype=float)
    c4_arr = np.array([r["cost_stems_grid_power"] for r in pol_recs], dtype=float)
    joint_ok = (c0_arr == 0) & (c2_arr == 0) & (c3_arr == 0) & (c4_arr == 0)
    out["joint_satisfaction_pct"] = float(100.0 * joint_ok.mean())

    # ----- C0 per-departure rates -----
    deps = np.array([r["ev_departure_departures"] for r in pol_recs], dtype=int)
    viol_counts = np.array(
        [r["ev_departure_violation_count_deficit"] for r in pol_recs], dtype=int
    )
    c0_costs = np.array([r["cost_ev_departure"] for r in pol_recs], dtype=float)
    total_deps = int(deps.sum())
    fixed_violators = int(viol_counts.sum())
    old_buggy = int(np.sum((deps > 0) & (c0_costs > 0)))

    out["c0_total_departures"] = total_deps
    out["c0_violator_departures_FIXED"] = fixed_violators
    out["c0_per_dep_rate_FIXED_pct"] = float(
        100.0 * fixed_violators / max(total_deps, 1)
    )
    out["c0_per_dep_rate_OLD_BUGGY_pct"] = float(
        100.0 * old_buggy / max(total_deps, 1)
    )

    # ----- Deficit kWh aggregates -----
    deficit_kwh = float(sum(r["ev_departure_deficit_kwh"] for r in pol_recs))
    avoidable_cost_units = float(sum(r["cost_ev_departure_avoidable"] for r in pol_recs))
    unavoidable_cost_units = float(sum(r["cost_ev_departure_unavoidable"] for r in pol_recs))
    out["total_deficit_kwh"] = deficit_kwh
    out["avoidable_kwh"] = avoidable_cost_units / max(ev_cost_scale, 1e-9)
    out["unavoidable_kwh"] = unavoidable_cost_units / max(ev_cost_scale, 1e-9)
    out["ev_cost_scale"] = ev_cost_scale

    # ----- Oracle metrics (window-sliced) -----
    pol_def_kwh = deficit_kwh
    orc_def_kwh = float(sum(r["ev_departure_deficit_kwh"] for r in orc_recs))
    avoid_wrt_oracle = max(0.0, pol_def_kwh - orc_def_kwh)
    out["policy_deficit_kwh"] = pol_def_kwh
    out["oracle_deficit_kwh"] = orc_def_kwh
    out["avoidable_wrt_oracle_kwh"] = avoid_wrt_oracle
    out["avoidable_wrt_oracle_pct"] = float(
        100.0 * avoid_wrt_oracle / pol_def_kwh if pol_def_kwh > 0.0 else 0.0
    )

    # ----- Per-step EV-action delta (mean |Δa| across timesteps) -----
    if len(pol_acts) >= 2:
        acts_arr = np.asarray(pol_acts, dtype=float)
        delta = np.abs(np.diff(acts_arr, axis=0)).mean()
        out["per_step_action_delta_mean"] = float(delta)
    else:
        out["per_step_action_delta_mean"] = 0.0

    # ----- District KPIs (full-year only; windows get null) -----
    kpi_keys = [
        "cost_total",
        "electricity_consumption_total",
        "carbon_emissions_total",
        "daily_peak_average",
        "all_time_peak_average",
        "ramping_average",
        "zero_net_energy",
        "daily_one_minus_load_factor_average",
    ]
    for cf in kpi_keys:
        full_key = f"District|{cf}"
        if is_full_year and full_key in cl_kpis_full_year:
            out[f"kpi_{cf}"] = float(cl_kpis_full_year[full_key])
        else:
            # CityLearn KPIs are episode-level; no defensible per-window value
            out[f"kpi_{cf}"] = None
    # Derived: daily_load_factor_avg = 1 - daily_one_minus_load_factor_average
    if out.get("kpi_daily_one_minus_load_factor_average") is not None:
        out["kpi_daily_load_factor_avg"] = (
            1.0 - out["kpi_daily_one_minus_load_factor_average"]
        )
    else:
        out["kpi_daily_load_factor_avg"] = None

    # ----- Per-building KPIs (full-year only; same caveat as district) -----
    # Stored as a nested dict {building_name: {cost_function: value}} so the
    # downstream JSON consumer can iterate buildings without parsing keys.
    per_building: Dict[str, Dict[str, float]] = {}
    if is_full_year:
        for full_key, val in cl_kpis_full_year.items():
            if "|" not in full_key:
                continue
            owner, cf = full_key.split("|", 1)
            if owner == "District":
                continue
            per_building.setdefault(owner, {})[cf] = float(val)
    out["kpi_per_building"] = per_building if per_building else None

    return out


# ---------------------------------------------------------------------------
# Baseline rollout (mirrors run_full_year_rollout but uses a baseline.act())
# ---------------------------------------------------------------------------

def run_full_year_rollout_baseline(env, baseline_actor, *, use_oracle=False,
                                   ev_action_indices=None, seed=0, max_steps=8760):
    """Run a baseline (rule-based controller) through the same rollout/recording
    pipeline as the trained policies."""
    from citylearn_safe.extractors import (
        unwrap_to_raw_citylearn_env, _oracle_connected_now,
        _build_ev_action_index_by_charger_id, _normalize_action_names,
    )
    try:
        obs, info = env.reset(seed=seed)
    except TypeError:
        obs, info = env.reset()

    raw_env = unwrap_to_raw_citylearn_env(env)
    oracle_idx_by_charger = {}
    if use_oracle:
        action_names = _normalize_action_names(getattr(raw_env, "action_names", []))
        oracle_idx_by_charger = _build_ev_action_index_by_charger_id(action_names)

    cl_env_for_meta = get_citylearn_env(env)
    n_buildings = len(cl_env_for_meta.buildings) if cl_env_for_meta is not None else 0
    charger_index: List[Tuple[int, int]] = []
    if cl_env_for_meta is not None:
        for bi, b in enumerate(cl_env_for_meta.buildings):
            chs = getattr(b, "electric_vehicle_chargers", None) or []
            for ci, _ in enumerate(chs):
                charger_index.append((bi, ci))
    n_chargers = len(charger_index)

    rewards, actions, info_records = [], [], []
    net_load_traj = np.full((max_steps, max(n_buildings, 1)), np.nan, dtype=np.float32)
    ev_soc_traj = np.full((max_steps, max(n_chargers, 1)), np.nan, dtype=np.float32)
    cl_kpis = {}
    for step in range(max_steps):
        action_np = baseline_actor.act(obs, info, env=env)
        action_np = np.asarray(action_np, dtype=np.float32)
        action_np = np.clip(action_np, env.action_space.low, env.action_space.high)

        if use_oracle and oracle_idx_by_charger:
            for cid, a_idx in oracle_idx_by_charger.items():
                if 0 <= a_idx < len(action_np) and _oracle_connected_now(raw_env, cid):
                    action_np[a_idx] = 1.0

        obs, reward, terminated, truncated, info = env.step(action_np)
        rewards.append(float(reward))
        actions.append(action_np.copy())

        if cl_env_for_meta is not None:
            ts = getattr(cl_env_for_meta, "time_step", step + 1)
            for bi in range(n_buildings):
                try:
                    nec = cl_env_for_meta.buildings[bi].net_electricity_consumption
                    if nec is not None and len(nec) > 0:
                        idx = min(ts - 1, len(nec) - 1)
                        if idx >= 0:
                            net_load_traj[step, bi] = float(nec[idx])
                except (AttributeError, IndexError, TypeError):
                    pass
            for k, (bi, ci) in enumerate(charger_index):
                try:
                    ch = cl_env_for_meta.buildings[bi].electric_vehicle_chargers[ci]
                    ev = getattr(ch, "connected_electric_vehicle", None)
                    if ev is not None:
                        soc_arr = ev.battery.soc
                        if soc_arr is not None and len(soc_arr) > 0:
                            idx = min(ts - 1, len(soc_arr) - 1)
                            if idx >= 0:
                                ev_soc_traj[step, k] = float(soc_arr[idx])
                except (AttributeError, IndexError, TypeError):
                    pass

        rec = {
            "ev_departure_departures": int(info.get("ev_departure_departures", 0)),
            "ev_departure_violation_count_deficit": int(
                info.get("ev_departure_violation_count_deficit", 0)),
            "building_power_violation_count": float(
                info.get("building_power_violation_count", 0.0)),
            "ev_departure_deficit_kwh": float(info.get("ev_departure_deficit_kwh", 0.0)),
            "ev_avoidable_deficit_kwh": float(info.get("ev_avoidable_deficit_kwh", 0.0)),
            "ev_unavoidable_deficit_kwh": float(info.get("ev_unavoidable_deficit_kwh", 0.0)),
            "cost_ev_departure_avoidable": float(info.get("cost_ev_departure_avoidable", 0.0)),
            "cost_ev_departure_unavoidable": float(info.get("cost_ev_departure_unavoidable", 0.0)),
        }
        for k in COST_KEYS:
            rec[k] = float(info.get(k, 0.0))
        info_records.append(rec)
        if terminated or truncated:
            break

    actual_len = len(rewards)
    net_load_traj = net_load_traj[:actual_len]
    ev_soc_traj = ev_soc_traj[:actual_len]

    cl_env = get_citylearn_env(env)
    if cl_env is not None and hasattr(cl_env, "evaluate"):
        try:
            kpi_df = cl_env.evaluate()
            if kpi_df is not None:
                for _, row in kpi_df.iterrows():
                    level = str(row.get("level", ""))
                    cf = str(row.get("cost_function", "unknown"))
                    bname = str(row.get("name", "unknown"))
                    v = row.get("value", None)
                    if v is None or (isinstance(v, float) and not np.isfinite(v)):
                        continue
                    if level == "district":
                        cl_kpis[f"District|{cf}"] = float(v)
                    else:
                        cl_kpis[f"{bname}|{cf}"] = float(v)
        except Exception as e:
            print(f"  [warn] KPI eval failed: {e}")

    return {"rewards": rewards, "actions": actions,
            "info_records": info_records, "cl_kpis": cl_kpis,
            "ep_len": actual_len,
            "net_load_traj": net_load_traj,
            "ev_soc_traj": ev_soc_traj,
            "charger_index": charger_index,
            "n_buildings": n_buildings}


def _evaluate_baseline_inner(name, cfg, device, *, eval_seed, applied, ev_cost_scale):
    """Inner: build env, instantiate baseline, run policy + oracle rollouts."""
    baseline_factory = cfg["baseline_factory"]
    actor = baseline_factory()

    env_pol = build_eval_env()
    ev_action_indices = find_ev_action_indices(env_pol)
    print(f"  [baseline {name}] obs_dim={env_pol.observation_space.shape[0]} "
          f"act_dim={env_pol.action_space.shape[0]} ev_idx={ev_action_indices}")

    t0 = time.time()
    policy_roll = run_full_year_rollout_baseline(
        env_pol, actor, use_oracle=False,
        ev_action_indices=ev_action_indices, seed=eval_seed,
    )
    print(f"  policy rollout: {policy_roll['ep_len']} steps in {time.time()-t0:.1f}s, "
          f"EpRet={sum(policy_roll['rewards']):.1f}")

    env_orc = build_eval_env()
    actor_orc = baseline_factory()  # Fresh state for oracle
    t0 = time.time()
    oracle_roll = run_full_year_rollout_baseline(
        env_orc, actor_orc, use_oracle=True,
        ev_action_indices=ev_action_indices, seed=eval_seed,
    )
    print(f"  oracle rollout: {oracle_roll['ep_len']} steps in {time.time()-t0:.1f}s")

    windows_out = {}
    for wname, wbounds in WINDOWS.items():
        windows_out[wname] = compute_window_metrics(
            policy_roll, oracle_roll, wbounds,
            ev_cost_scale=ev_cost_scale,
            cl_kpis_full_year=policy_roll["cl_kpis"],
            is_full_year=(wname == "full_year"),
        )

    return {
        "name": name,
        "ckpt_path": "BASELINE",
        "actor_type": "baseline_ext",
        "saute_pad": False,
        "footnote": cfg.get("footnote"),
        "training_seed": None,
        "env_vars_hash": env_vars_hash(applied),
        "ev_cost_scale": ev_cost_scale,
        "windows": windows_out,
        # In-memory per-step trajectories (NOT serialised to JSON; written
        # separately as .npy by the main loop).
        "_per_step": {
            "actions": np.asarray(policy_roll["actions"], dtype=np.float32),
            "net_load": policy_roll["net_load_traj"],
            "ev_soc": policy_roll["ev_soc_traj"],
            "charger_index": policy_roll["charger_index"],
            "costs": np.asarray(
                [[r.get(k, 0.0) for k in COST_KEYS]
                 for r in policy_roll["info_records"]],
                dtype=np.float32,
            ),
        },
    }


# ---------------------------------------------------------------------------
# Per-model orchestration
# ---------------------------------------------------------------------------

def resolve_ckpt_path(cfg: Dict[str, Any]) -> str | None:
    """Resolve `ckpt` (literal) or `ckpt_glob` (glob with single match).
    Returns sentinel "BASELINE" for baseline configs (no ckpt needed)."""
    if cfg.get("actor_type") == "baseline_ext":
        return "BASELINE"
    if "ckpt" in cfg:
        path = os.path.join(PROJECT_ROOT, cfg["ckpt"])
        return path if os.path.isfile(path) else None
    if "ckpt_glob" in cfg:
        import glob
        pattern = os.path.join(PROJECT_ROOT, cfg["ckpt_glob"])
        matches = sorted(glob.glob(pattern))
        return matches[-1] if matches else None
    return None


def evaluate_one_model(
    name: str,
    cfg: Dict[str, Any],
    device: torch.device,
    *,
    eval_seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate one model across all 3 windows. Env-var isolated."""
    with env_snapshot():
        # Apply COMMON_ENV + model overrides
        applied: Dict[str, str] = {}
        for k, v in COMMON_ENV.items():
            os.environ[k] = v
            applied[k] = v
        for k, v in cfg.get("extra_env", {}).items():
            os.environ[k] = v
            applied[k] = v
        schema_path = os.path.join(
            PROJECT_ROOT,
            "data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json",
        )
        os.environ["CITYLEARN_SCHEMA"] = schema_path
        applied["CITYLEARN_SCHEMA"] = schema_path

        ev_cost_scale = float(os.environ.get("CITYLEARN_EV_COST_SCALE", "1.0"))

        ckpt_path = resolve_ckpt_path(cfg)
        if ckpt_path is None:
            raise FileNotFoundError(
                f"Checkpoint not found for {name}: {cfg.get('ckpt') or cfg.get('ckpt_glob')}"
            )

        actor_type = cfg["actor_type"]
        if actor_type == "baseline_ext":
            return _evaluate_baseline_inner(name, cfg, device, eval_seed=eval_seed,
                                            applied=applied, ev_cost_scale=ev_cost_scale)

        # Build env once for policy rollout
        env_pol = build_eval_env()
        obs_dim = int(env_pol.observation_space.shape[0])
        act_dim = int(env_pol.action_space.shape[0])
        model, obs_norm, saute_pad = load_actor_from_ckpt(
            ckpt_path, actor_type, obs_dim, act_dim, device
        )

        ev_action_indices = find_ev_action_indices(env_pol)

        print(f"  obs_dim={obs_dim} act_dim={act_dim} ev_idx={ev_action_indices} "
              f"saute_pad={saute_pad} ev_cost_scale={ev_cost_scale}")

        # Policy rollout (full year)
        t0 = time.time()
        policy_roll = run_full_year_rollout(
            env_pol, model, obs_norm, saute_pad, device,
            use_oracle=False, ev_action_indices=ev_action_indices,
            seed=eval_seed,
        )
        print(f"  policy rollout: {policy_roll['ep_len']} steps in {time.time()-t0:.1f}s, "
              f"EpRet={sum(policy_roll['rewards']):.1f}")

        # Oracle rollout (fresh env so nothing carries over)
        env_orc = build_eval_env()
        t0 = time.time()
        oracle_roll = run_full_year_rollout(
            env_orc, model, obs_norm, saute_pad, device,
            use_oracle=True, ev_action_indices=ev_action_indices,
            seed=eval_seed,
        )
        print(f"  oracle rollout: {oracle_roll['ep_len']} steps in {time.time()-t0:.1f}s")

        # Compute metrics for each window
        windows_out: Dict[str, Dict[str, Any]] = {}
        for wname, wbounds in WINDOWS.items():
            windows_out[wname] = compute_window_metrics(
                policy_roll, oracle_roll, wbounds,
                ev_cost_scale=ev_cost_scale,
                cl_kpis_full_year=policy_roll["cl_kpis"],
                is_full_year=(wname == "full_year"),
            )

        return {
            "name": name,
            "ckpt_path": os.path.relpath(ckpt_path, PROJECT_ROOT),
            "actor_type": actor_type,
            "saute_pad": saute_pad,
            "footnote": cfg.get("footnote"),
            "training_seed": cfg.get("training_seed"),
            "env_vars_hash": env_vars_hash(applied),
            "ev_cost_scale": ev_cost_scale,
            "windows": windows_out,
            # In-memory per-step trajectories (NOT serialised to JSON; written
            # separately as .npy by the main loop).
            "_per_step": {
                "actions": np.asarray(policy_roll["actions"], dtype=np.float32),
                "net_load": policy_roll["net_load_traj"],
                "ev_soc": policy_roll["ev_soc_traj"],
                "charger_index": policy_roll["charger_index"],
                "costs": np.asarray(
                    [[r.get(k, 0.0) for k in COST_KEYS]
                     for r in policy_roll["info_records"]],
                    dtype=np.float32,
                ),
            },
        }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

# Stable column order for CSV (one row per algo x window)
CSV_COLUMNS = [
    "algo", "window", "window_start", "window_end", "window_steps",
    "ckpt_path", "training_seed", "env_vars_hash", "eval_date", "git_sha",
    "ep_ret", "ep_cost_total",
    "c0_total_departures", "c0_violator_departures_FIXED",
    "c0_per_dep_rate_FIXED_pct", "c0_per_dep_rate_OLD_BUGGY_pct",
    "total_deficit_kwh", "avoidable_kwh", "unavoidable_kwh",
    "policy_deficit_kwh", "oracle_deficit_kwh",
    "avoidable_wrt_oracle_kwh", "avoidable_wrt_oracle_pct",
    "viol_pct_C0", "viol_pct_C1", "viol_pct_C2", "viol_pct_C3", "viol_pct_C4",
    "cost_C0_total", "cost_C1_total", "cost_C2_total", "cost_C3_total", "cost_C4_total",
    "per_step_action_delta_mean",
    "kpi_cost_total", "kpi_electricity_consumption_total",
    "kpi_carbon_emissions_total", "kpi_daily_peak_average",
    "kpi_all_time_peak_average", "kpi_ramping_average",
    "kpi_zero_net_energy", "kpi_daily_load_factor_avg",
    "footnote",
]


def write_csv(results: Dict[str, Any], path: str, eval_date: str, git_sha: str) -> None:
    with open(path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for algo, info in results.items():
            if "error" in info:
                continue
            for win_name, win_metrics in info["windows"].items():
                row = {c: "" for c in CSV_COLUMNS}
                row["algo"] = algo
                row["window"] = win_name
                row["ckpt_path"] = info.get("ckpt_path", "")
                row["training_seed"] = info.get("training_seed", "")
                row["env_vars_hash"] = info.get("env_vars_hash", "")
                row["eval_date"] = eval_date
                row["git_sha"] = git_sha
                row["footnote"] = info.get("footnote") or ""
                for k, v in win_metrics.items():
                    if k in CSV_COLUMNS:
                        row[k] = v if v is not None else ""
                w.writerow(row)


class TeeWriter:
    """Mirror writes to both stdout and a log file."""

    def __init__(self, path: str):
        self.fp = open(path, "w", buffering=1)
        self.stdout = sys.stdout

    def write(self, s: str) -> int:
        self.stdout.write(s)
        self.fp.write(s)
        return len(s)

    def flush(self) -> None:
        self.stdout.flush()
        self.fp.flush()

    def close(self) -> None:
        try:
            self.fp.close()
        except Exception:
            pass


def get_git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", PROJECT_ROOT, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def print_summary_table(results: Dict[str, Any]) -> None:
    """Compact summary (full_year window only) for stdout/log."""
    print()
    print("=" * 130)
    print("  BENCHMARK SUMMARY (full_year window)")
    print("=" * 130)
    hdr = (
        f"  {'algo':<20s} | {'EpRet':>9s} | {'EpCost':>9s} | "
        f"{'C0%FIX':>7s} | {'C0%OLD':>7s} | "
        f"{'pol_kWh':>9s} | {'orc_kWh':>9s} | {'avoid%':>7s}"
    )
    print(hdr)
    print("  " + "-" * 124)
    for algo, info in results.items():
        if "error" in info:
            print(f"  {algo:<20s} | ERROR: {info['error']}")
            continue
        m = info["windows"]["full_year"]
        print(
            f"  {algo:<20s} | {m['ep_ret']:>9.1f} | {m['ep_cost_total']:>9.0f} | "
            f"{m['c0_per_dep_rate_FIXED_pct']:>6.1f}% | {m['c0_per_dep_rate_OLD_BUGGY_pct']:>6.1f}% | "
            f"{m['policy_deficit_kwh']:>9.1f} | {m['oracle_deficit_kwh']:>9.1f} | "
            f"{m['avoidable_wrt_oracle_pct']:>6.1f}%"
        )
    print("=" * 130)
    print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-csac", action="store_true",
                        help="Skip CSAC_LB_R28reward (e.g. if still training).")
    parser.add_argument("--only", action="append", default=None,
                        help="Evaluate only the given algo name(s). Repeatable.")
    parser.add_argument("--seed", type=int, default=42, help="Eval seed.")
    parser.add_argument("--no-baselines", action="store_true",
                        help="Skip RBC / charge-on-arrival / zero-action baselines.")
    parser.add_argument("--out-dir", default=None,
                        help="Output dir (default: scripts/eval_outputs).")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(PROJECT_ROOT, "scripts", "eval_outputs")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(out_dir, f"eval_r28_benchmark_FINAL_{ts}.log")
    json_path = os.path.join(out_dir, f"eval_r28_benchmark_FINAL_{ts}.json")
    csv_path = os.path.join(out_dir, f"eval_r28_benchmark_FINAL_{ts}.csv")
    per_step_root = os.path.join(out_dir, "per_step_data")
    os.makedirs(per_step_root, exist_ok=True)

    tee = TeeWriter(log_path)
    sys.stdout = tee  # capture all prints

    eval_date = datetime.now().isoformat(timespec="seconds")
    git_sha = get_git_sha()

    print(f"=== eval_r28_benchmark.py ===")
    print(f"timestamp:  {ts}")
    print(f"eval_date:  {eval_date}")
    print(f"git_sha:    {git_sha}")
    print(f"log_path:   {log_path}")
    print(f"json_path:  {json_path}")
    print(f"csv_path:   {csv_path}")
    print(f"windows:    {WINDOWS}")
    print()

    # Load baseline controllers (RBC, charge-on-arrival, zero-action)
    if not args.no_baselines:
        _load_baselines()

    # Pick which models to run
    selected = list(MODELS.keys())
    if args.only:
        # Allow case-flexible matching
        wanted = [s.strip() for s in args.only]
        selected = [n for n in selected if n in wanted]
        if not selected:
            print(f"[error] --only filter matched no models. Available: {list(MODELS)}")
            return 2
    if args.skip_csac and "CSAC_LB_R28reward" in selected:
        selected.remove("CSAC_LB_R28reward")
        print("[info] Skipping CSAC_LB_R28reward (--skip-csac)")

    # Auto-skip if a non-CSAC checkpoint is missing
    for name in list(selected):
        cfg = MODELS[name]
        ckpt = resolve_ckpt_path(cfg)
        if ckpt is None:
            if name == "CSAC_LB_R28reward":
                print(f"[info] {name} ckpt not yet present -> auto-skip")
                selected.remove(name)
            else:
                print(f"[warn] {name} ckpt missing at {cfg.get('ckpt') or cfg.get('ckpt_glob')} - will fail")

    print(f"\nEvaluating {len(selected)} models: {selected}\n")

    device = torch.device("cpu")
    results: Dict[str, Any] = {}

    for name in selected:
        cfg = MODELS[name]
        print(f"\n{'='*70}\n  {name}\n  ckpt: {cfg.get('ckpt') or cfg.get('ckpt_glob')}\n{'='*70}")
        t0 = time.time()
        try:
            results[name] = evaluate_one_model(name, cfg, device, eval_seed=args.seed)
            results[name]["eval_date"] = eval_date
            results[name]["git_sha"] = git_sha
            elapsed = time.time() - t0
            m = results[name]["windows"]["full_year"]
            print(
                f"  DONE in {elapsed:.1f}s | "
                f"EpRet={m['ep_ret']:.1f} | "
                f"C0_FIX={m['c0_per_dep_rate_FIXED_pct']:.1f}% | "
                f"pol_kWh={m['policy_deficit_kwh']:.1f} | "
                f"orc_kWh={m['oracle_deficit_kwh']:.1f}"
            )

            # Persist per-step trajectories as .npy (one folder per algo).
            per_step = results[name].pop("_per_step", None)
            if per_step is not None:
                algo_dir = os.path.join(per_step_root, name)
                os.makedirs(algo_dir, exist_ok=True)
                np.save(os.path.join(algo_dir, "actions.npy"), per_step["actions"])
                np.save(os.path.join(algo_dir, "net_load.npy"), per_step["net_load"])
                np.save(os.path.join(algo_dir, "ev_soc.npy"), per_step["ev_soc"])
                np.save(os.path.join(algo_dir, "costs.npy"), per_step["costs"])
                # Charger->building map so downstream plotting can label axes.
                with open(os.path.join(algo_dir, "charger_index.json"), "w") as fp:
                    json.dump(
                        [{"building_idx": bi, "charger_idx": ci}
                         for (bi, ci) in per_step["charger_index"]],
                        fp, indent=2,
                    )
                print(f"  per-step .npy written: {algo_dir}")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  FAILED in {elapsed:.1f}s: {type(e).__name__}: {e}")
            traceback.print_exc()
            results[name] = {"error": f"{type(e).__name__}: {e}"}

    # Defensive: drop any leftover _per_step blobs (e.g. on failed algos).
    for v in results.values():
        if isinstance(v, dict):
            v.pop("_per_step", None)

    # Persist outputs
    with open(json_path, "w") as fp:
        json.dump(results, fp, indent=2, default=str)
    print(f"\nJSON written: {json_path}")
    print(f"Per-step root: {per_step_root}")

    write_csv(results, csv_path, eval_date, git_sha)
    print(f"CSV written:  {csv_path}")

    print_summary_table(results)

    print("\nFootnotes:")
    print("  - PPOSaute uses a constant safety_state=1.0 padded onto obs (biased proxy).")
    print("  - District KPIs are episode-level; per-window kpi_* fields are null on purpose.")
    print("  - C0 FIXED rate uses info[ev_departure_violation_count_deficit] (per departure).")
    print("    OLD BUGGY rate counts steps with any deficit; under-counts when 2+ EVs depart same hour.")

    sys.stdout = tee.stdout
    tee.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
