#!/usr/bin/env python3
"""Evaluate 4 'dumb' RBC variants alongside existing baselines and CSAC-LB.

Dumb RBC variants:
  1. Fixed-output:   always cools at 50% regardless of temperature
  2. Bang-bang:       full ON when tin > 25, full OFF otherwise (oscillation)
  3. No-schedule:     always max cooling (100%) — extreme waste
  4. Delayed-response: reacts only when tin > 28 (2°C late), proportional
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from gymnasium.spaces import Box

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
VENDORED_OMNISAFE = PROJECT_ROOT / "vendor_deps"
if str(VENDORED_OMNISAFE) not in sys.path:
    sys.path.insert(0, str(VENDORED_OMNISAFE))

from citylearn_safe.omni_env_temp_cooling_only import (
    CityLearnTempCoolingOnlyMaskedRewardCMDP,
)
from omnisafe.models.actor.gaussian_sac_actor import GaussianSACActor


# ── Dumb RBC action generators ──────────────────────────────────────────────

def _rbc_fixed_output(env: CityLearnTempCoolingOnlyMaskedRewardCMDP) -> np.ndarray:
    """Always cool at 50% regardless of temperature."""
    return np.full(env.cooling_action_dim, 0.5, dtype=np.float32)


def _rbc_bangbang(env: CityLearnTempCoolingOnlyMaskedRewardCMDP) -> np.ndarray:
    """Full ON when any building tin > 25, full OFF otherwise."""
    actions = np.zeros(env.cooling_action_dim, dtype=np.float32)
    for i in range(env.cooling_action_dim):
        tin, tset = env._cooling_state(i)
        if tin is None or not np.isfinite(tin):
            actions[i] = 0.0
            continue
        actions[i] = 1.0 if tin > 25.0 else 0.0
    return actions


def _rbc_always_max(env: CityLearnTempCoolingOnlyMaskedRewardCMDP) -> np.ndarray:
    """Always cool at 100% — maximum waste."""
    return np.ones(env.cooling_action_dim, dtype=np.float32)


def _rbc_delayed(env: CityLearnTempCoolingOnlyMaskedRewardCMDP) -> np.ndarray:
    """Delayed response: only reacts when tin > 28°C (2°C too late), proportional."""
    actions = np.zeros(env.cooling_action_dim, dtype=np.float32)
    for i in range(env.cooling_action_dim):
        tin, tset = env._cooling_state(i)
        if tin is None or not np.isfinite(tin):
            actions[i] = 0.0
            continue
        threshold = 28.0  # 2°C above upper comfort bound
        if tin > threshold:
            error = tin - threshold
            actions[i] = float(min(1.0, error / 3.0))
        else:
            actions[i] = 0.0
    return actions


# ── Reused from evaluate_csaclb_temp_case_study.py ──────────────────────────

@dataclass
class RolloutResult:
    name: str
    total_reward: float
    total_cost: float
    violation_rate: float
    discomfort_rate: float
    mean_cooling_action: float
    reward_economic: float
    reward_stability: float
    reward_renewable: float
    reward_comfort: float
    district_import_kwh: float
    district_net_kwh: float
    total_steps: int
    active_steps: int
    comfort_violations: int
    discomfort_count: float
    occupied_count: float
    kpis: dict[str, float]


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, encoding="utf-8") as f:
        return json.load(f)


def _apply_env_overrides(run_dir: Path) -> None:
    cfg = _load_run_config(run_dir)
    for key, value in (cfg.get("env_overrides") or {}).items():
        os.environ[str(key)] = str(value)


def _selected_kpis(eval_df: pd.DataFrame) -> dict[str, float]:
    if eval_df is None or eval_df.empty:
        return {}
    cols = {str(c).lower(): c for c in eval_df.columns}
    key_col = cols.get("cost_function")
    value_col = cols.get("value")
    if key_col is None or value_col is None:
        return {}
    out: dict[str, float] = {}
    for _, row in eval_df.iterrows():
        try:
            out[str(row[key_col])] = float(row[value_col])
        except Exception:
            continue
    return out


def _evaluate_citylearn_kpis(env) -> dict[str, float]:
    city = env._get_citylearn()
    if city is None or not hasattr(city, "evaluate"):
        return {}
    try:
        eval_df = city.evaluate()
    except Exception:
        return {}
    return _selected_kpis(eval_df)


def _make_eval_env(run_dir: Path):
    _apply_env_overrides(run_dir)
    return CityLearnTempCoolingOnlyMaskedRewardCMDP("CityLearnTemp-CoolingOnly-CSACLB-v0")


def _load_actor(checkpoint: Path) -> GaussianSACActor:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    pi_state = state["pi"]
    obs_dim = int(pi_state["net.0.weight"].shape[1])
    act_dim = int(pi_state["net.4.weight"].shape[0] // 2)
    hidden = []
    layer_idx = 0
    while f"net.{layer_idx}.weight" in pi_state:
        weight = pi_state[f"net.{layer_idx}.weight"]
        if layer_idx + 2 < 5:
            hidden.append(int(weight.shape[0]))
        layer_idx += 2
        if f"net.{layer_idx}.weight" not in pi_state:
            break
    if len(hidden) < 2:
        hidden = [256, 256]

    actor = GaussianSACActor(
        obs_space=Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
        act_space=Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32),
        hidden_sizes=hidden,
        activation="relu",
        weight_initialization_mode="kaiming_uniform",
    )
    actor.load_state_dict(pi_state, strict=True)
    actor.eval()
    return actor


def _tanh_to_unit_interval(a: np.ndarray) -> np.ndarray:
    return np.clip((a + 1.0) * 0.5, 0.0, 1.0)


# ── Main rollout function ───────────────────────────────────────────────────

POLICY_MAP = {
    "zero": lambda env: np.zeros(env.cooling_action_dim, dtype=np.float32),
    "rbc": lambda env: env._rbc_cooling_action(),
    "rbc2": lambda env: env._rbc2_cooling_action(),
    "rbc3": lambda env: env._rbc3_cooling_action(),
    "fixed_output": _rbc_fixed_output,
    "bangbang": _rbc_bangbang,
    "always_max": _rbc_always_max,
    "delayed": _rbc_delayed,
}


def run_rollout(
    name: str,
    policy_kind: str,
    seed: int,
    run_dir: Path,
    checkpoint: Path | None = None,
) -> RolloutResult:
    env = _make_eval_env(run_dir)
    actor = _load_actor(checkpoint) if checkpoint is not None else None

    obs, _ = env.reset(seed=seed)
    terminated = truncated = False

    total_reward = total_cost = 0.0
    total_steps = active_steps = comfort_violations = 0
    occupied_count = discomfort_count = action_sum = 0.0
    action_count = 0
    reward_economic = reward_stability = reward_renewable = reward_comfort = 0.0
    district_import_kwh = district_net_kwh = 0.0

    while not (bool(terminated) or bool(truncated)):
        if policy_kind == "ckpt":
            assert actor is not None
            obs_t = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
            with torch.no_grad():
                raw = actor.predict(obs_t, deterministic=True).cpu().numpy().reshape(-1)
            action = _tanh_to_unit_interval(raw)
        else:
            action = POLICY_MAP[policy_kind](env)

        obs, reward, cost, terminated, truncated, info = env.step(action)
        total_steps += 1
        total_reward += float(reward)
        total_cost += float(cost)
        if float(info.get("comfort_in_warmup", 0.0)) < 0.5:
            active_steps += 1
            comfort_violations += int(float(info.get("comfort_violation", 0.0)) > 0.5)
        occupied_count += float(info.get("occupied_count", 0.0))
        discomfort_count += float(info.get("discomfort_count", 0.0))
        action_sum += float(np.sum(action))
        action_count += int(np.size(action))
        reward_economic += float(info.get("reward_economic", 0.0))
        reward_stability += float(info.get("reward_stability", 0.0))
        reward_renewable += float(info.get("reward_renewable", 0.0))
        reward_comfort += float(info.get("reward_comfort", 0.0))
        district_import_kwh += float(info.get("district_import_kwh", 0.0))
        district_net_kwh += float(info.get("district_net_kwh", 0.0))

    kpis = _evaluate_citylearn_kpis(env)
    return RolloutResult(
        name=name,
        total_reward=total_reward,
        total_cost=total_cost,
        violation_rate=(comfort_violations / active_steps) if active_steps > 0 else 0.0,
        discomfort_rate=(discomfort_count / occupied_count) if occupied_count > 0 else 0.0,
        mean_cooling_action=(action_sum / action_count) if action_count > 0 else 0.0,
        reward_economic=reward_economic,
        reward_stability=reward_stability,
        reward_renewable=reward_renewable,
        reward_comfort=reward_comfort,
        district_import_kwh=district_import_kwh,
        district_net_kwh=district_net_kwh,
        total_steps=total_steps,
        active_steps=active_steps,
        comfort_violations=comfort_violations,
        discomfort_count=discomfort_count,
        occupied_count=occupied_count,
        kpis=kpis,
    )


def main():
    RUN_DIR = Path(
        "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/runs"
        "/csac_lb_temp_strict_40ep"
        "/CSACLBTemp-{CityLearnTemp-CoolingOnly-CSACLB-v0}"
        "/seed-000-2026-03-27-23-30-03"
    )
    SEED = 0

    # Find best/final epoch
    progress = pd.read_csv(RUN_DIR / "progress.csv")
    best_epoch = int(float(progress.loc[progress["Metrics/TestEpCost"].astype(float).idxmin(), "Train/Epoch"]))
    final_epoch = max(
        int(p.stem.split("-")[1])
        for p in (RUN_DIR / "torch_save").glob("epoch-*.pt")
    )
    best_ckpt = RUN_DIR / "torch_save" / f"epoch-{best_epoch}.pt"
    final_ckpt = RUN_DIR / "torch_save" / f"epoch-{final_epoch}.pt"

    print(f"Best epoch: {best_epoch}, Final epoch: {final_epoch}")
    print(f"Run dir: {RUN_DIR}\n")

    evaluations = [
        ("CSAC-LB (best)", "ckpt", best_ckpt),
        ("CSAC-LB (final)", "ckpt", final_ckpt),
        ("Cooling RBC", "rbc", None),
        ("RBC 2 (weak)", "rbc2", None),
        ("RBC 3 (aggressive)", "rbc3", None),
        ("Zero (no cooling)", "zero", None),
        ("Dumb: Fixed 50%", "fixed_output", None),
        ("Dumb: Bang-Bang", "bangbang", None),
        ("Dumb: Always Max", "always_max", None),
        ("Dumb: Delayed (28°C)", "delayed", None),
    ]

    rows = []
    for idx, (name, kind, ckpt) in enumerate(evaluations):
        print(f"[{idx+1}/{len(evaluations)}] Evaluating: {name} ...", flush=True)
        r = run_rollout(name, kind, SEED, RUN_DIR, ckpt)

        row = {
            "name": r.name,
            "violation_rate": r.violation_rate,
            "discomfort_rate": r.discomfort_rate,
            "mean_cooling_action": r.mean_cooling_action,
            "total_reward": r.total_reward,
            "total_cost": r.total_cost,
            "reward_economic": r.reward_economic,
            "reward_stability": r.reward_stability,
            "reward_renewable": r.reward_renewable,
            "reward_comfort": r.reward_comfort,
        }
        row.update({f"kpi::{k}": v for k, v in r.kpis.items()})
        rows.append(row)

        # Print summary as we go
        print(f"   Violation: {r.violation_rate:.3f}  |  "
              f"Cost KPI: {r.kpis.get('cost_total', -1):.3f}  |  "
              f"Consumption KPI: {r.kpis.get('electricity_consumption_total', -1):.3f}  |  "
              f"Ramping KPI: {r.kpis.get('ramping_average', -1):.3f}  |  "
              f"Mean cool: {r.mean_cooling_action:.3f}")

    # Save JSON
    out_path = RUN_DIR / "eval_dumb_rbcs.json"
    with open(out_path, "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Print comparison table
    print("\n" + "=" * 120)
    print(f"{'Method':<25} {'Violation':>10} {'Discomfort':>11} {'Cost KPI':>10} "
          f"{'Emis KPI':>10} {'Cons KPI':>10} {'Ramp KPI':>10} {'Mean Cool':>10}")
    print("-" * 120)
    for row in rows:
        print(f"{row['name']:<25} "
              f"{row['violation_rate']:>10.3f} "
              f"{row['discomfort_rate']:>11.3f} "
              f"{row.get('kpi::cost_total', -1):>10.3f} "
              f"{row.get('kpi::carbon_emissions_total', -1):>10.3f} "
              f"{row.get('kpi::electricity_consumption_total', -1):>10.3f} "
              f"{row.get('kpi::ramping_average', -1):>10.3f} "
              f"{row['mean_cooling_action']:>10.3f}")
    print("=" * 120)

    # Highlight: which baselines does CSAC-LB beat on ALL metrics?
    csac_final = next(r for r in rows if r["name"] == "CSAC-LB (final)")
    print("\n>>> CSAC-LB (final) dominates these baselines on ALL key metrics:")
    key_kpis = ["kpi::cost_total", "kpi::carbon_emissions_total",
                "kpi::electricity_consumption_total", "kpi::ramping_average"]
    for row in rows:
        if row["name"].startswith("CSAC-LB"):
            continue
        beats_violation = csac_final["violation_rate"] < row["violation_rate"]
        beats_all_kpis = all(
            csac_final.get(k, 999) <= row.get(k, 999) for k in key_kpis
        )
        status = "DOMINATED" if (beats_violation and beats_all_kpis) else "not dominated"
        print(f"   {row['name']:<25} -> {status}")


if __name__ == "__main__":
    main()
