#!/usr/bin/env python3
"""Final evaluation: CSAC-LB vs new Bang-Bang RBC baseline + other variants."""
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

from citylearn_safe.omni_env_temp_cooling_only import CityLearnTempCoolingOnlyMaskedRewardCMDP
from omnisafe.models.actor.gaussian_sac_actor import GaussianSACActor


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
    with open(cfg_path) as f:
        return json.load(f)

def _apply_env_overrides(run_dir: Path):
    cfg = _load_run_config(run_dir)
    for key, value in (cfg.get("env_overrides") or {}).items():
        os.environ[str(key)] = str(value)

def _selected_kpis(eval_df):
    if eval_df is None or eval_df.empty:
        return {}
    cols = {str(c).lower(): c for c in eval_df.columns}
    key_col = cols.get("cost_function")
    value_col = cols.get("value")
    if key_col is None or value_col is None:
        return {}
    out = {}
    for _, row in eval_df.iterrows():
        try:
            out[str(row[key_col])] = float(row[value_col])
        except Exception:
            continue
    return out

def _evaluate_kpis(env):
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

def _tanh_to_unit_interval(a):
    return np.clip((a + 1.0) * 0.5, 0.0, 1.0)


def run_rollout(name, policy_kind, seed, run_dir, checkpoint=None):
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
            obs_t = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
            with torch.no_grad():
                raw = actor.predict(obs_t, deterministic=True).cpu().numpy().reshape(-1)
            action = _tanh_to_unit_interval(raw)
        elif policy_kind == "rbc":
            action = env._rbc_cooling_action()
        elif policy_kind == "rbc2":
            action = env._rbc2_cooling_action()
        elif policy_kind == "rbc3":
            action = env._rbc3_cooling_action()
        elif policy_kind == "zero":
            action = np.zeros(env.cooling_action_dim, dtype=np.float32)
        else:
            raise ValueError(f"Unknown: {policy_kind}")

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

    kpis = _evaluate_kpis(env)
    return RolloutResult(
        name=name, total_reward=total_reward, total_cost=total_cost,
        violation_rate=(comfort_violations / active_steps) if active_steps > 0 else 0.0,
        discomfort_rate=(discomfort_count / occupied_count) if occupied_count > 0 else 0.0,
        mean_cooling_action=(action_sum / action_count) if action_count > 0 else 0.0,
        reward_economic=reward_economic, reward_stability=reward_stability,
        reward_renewable=reward_renewable, reward_comfort=reward_comfort,
        district_import_kwh=district_import_kwh, district_net_kwh=district_net_kwh,
        total_steps=total_steps, active_steps=active_steps,
        comfort_violations=comfort_violations, discomfort_count=discomfort_count,
        occupied_count=occupied_count, kpis=kpis,
    )


def main():
    RUN_DIR = Path(
        "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/runs"
        "/csac_lb_temp_strict_40ep"
        "/CSACLBTemp-{CityLearnTemp-CoolingOnly-CSACLB-v0}"
        "/seed-000-2026-03-27-23-30-03"
    )
    SEED = 0

    progress = pd.read_csv(RUN_DIR / "progress.csv")
    best_epoch = int(float(progress.loc[progress["Metrics/TestEpCost"].astype(float).idxmin(), "Train/Epoch"]))
    final_epoch = max(int(p.stem.split("-")[1]) for p in (RUN_DIR / "torch_save").glob("epoch-*.pt"))
    best_ckpt = RUN_DIR / "torch_save" / f"epoch-{best_epoch}.pt"
    final_ckpt = RUN_DIR / "torch_save" / f"epoch-{final_epoch}.pt"

    print(f"Best epoch: {best_epoch}, Final epoch: {final_epoch}\n")

    evaluations = [
        ("CSAC-LB best", "ckpt", best_ckpt),
        ("CSAC-LB final", "ckpt", final_ckpt),
        ("Cooling RBC", "rbc", None),          # Now bang-bang at 24°C
        ("RBC 2", "rbc2", None),
        ("RBC 3", "rbc3", None),
        ("Zero", "zero", None),
    ]

    rows = []
    for idx, (name, kind, ckpt) in enumerate(evaluations):
        print(f"[{idx+1}/{len(evaluations)}] {name} ...", flush=True)
        r = run_rollout(name, kind, SEED, RUN_DIR, ckpt)
        row = {
            "name": r.name,
            "total_reward": r.total_reward, "total_cost": r.total_cost,
            "violation_rate": r.violation_rate, "discomfort_rate": r.discomfort_rate,
            "mean_cooling_action": r.mean_cooling_action,
            "reward_economic": r.reward_economic, "reward_stability": r.reward_stability,
            "reward_renewable": r.reward_renewable, "reward_comfort": r.reward_comfort,
            "district_import_kwh": r.district_import_kwh, "district_net_kwh": r.district_net_kwh,
            "total_steps": r.total_steps, "active_steps": r.active_steps,
            "comfort_violations": r.comfort_violations,
            "discomfort_count": r.discomfort_count, "occupied_count": r.occupied_count,
        }
        row.update({f"kpi::{k}": v for k, v in r.kpis.items()})
        rows.append(row)
        print(f"   Viol={r.violation_rate:.3f}  Cost={r.kpis.get('cost_total',-1):.3f}  "
              f"Cons={r.kpis.get('electricity_consumption_total',-1):.3f}  "
              f"Ramp={r.kpis.get('ramping_average',-1):.3f}  "
              f"Emis={r.kpis.get('carbon_emissions_total',-1):.3f}  "
              f"Peak={r.kpis.get('daily_peak_average',-1):.3f}  "
              f"MeanCool={r.mean_cooling_action:.3f}")

    # Save
    out_path = RUN_DIR / "eval_case_study_bangbang.json"
    payload = {"best_epoch": best_epoch, "final_epoch": final_epoch, "rows": rows}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary table
    print("\n" + "=" * 130)
    print(f"{'Method':<20} {'Violation':>9} {'Discomf':>9} {'Cost':>9} {'Emis':>9} "
          f"{'Cons':>9} {'Peak':>9} {'Ramp':>9} {'MeanCool':>9}")
    print("-" * 130)
    for row in rows:
        print(f"{row['name']:<20} "
              f"{row['violation_rate']:>9.3f} "
              f"{row['discomfort_rate']:>9.3f} "
              f"{row.get('kpi::cost_total', -1):>9.3f} "
              f"{row.get('kpi::carbon_emissions_total', -1):>9.3f} "
              f"{row.get('kpi::electricity_consumption_total', -1):>9.3f} "
              f"{row.get('kpi::daily_peak_average', -1):>9.3f} "
              f"{row.get('kpi::ramping_average', -1):>9.3f} "
              f"{row['mean_cooling_action']:>9.3f}")
    print("=" * 130)

    # Dominance check
    csac_best = next(r for r in rows if r["name"] == "CSAC-LB best")
    csac_final = next(r for r in rows if r["name"] == "CSAC-LB final")
    kpi_keys = ["kpi::cost_total", "kpi::carbon_emissions_total",
                "kpi::electricity_consumption_total", "kpi::ramping_average",
                "kpi::daily_peak_average"]

    for csac_row, label in [(csac_best, "CSAC-LB best"), (csac_final, "CSAC-LB final")]:
        print(f"\n>>> {label} dominance check:")
        for row in rows:
            if row["name"].startswith("CSAC-LB"):
                continue
            beats_viol = csac_row["violation_rate"] < row["violation_rate"]
            beats_kpis = all(csac_row.get(k, 999) <= row.get(k, 999) for k in kpi_keys)
            print(f"   vs {row['name']:<20} Viol: {'WIN' if beats_viol else 'LOSE'}  "
                  f"KPIs: {'ALL WIN' if beats_kpis else 'some lose'}")


if __name__ == "__main__":
    main()
