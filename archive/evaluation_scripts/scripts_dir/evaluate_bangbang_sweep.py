#!/usr/bin/env python3
"""Sweep bang-bang threshold to find where cost/consumption KPIs exceed CSAC-LB."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

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


def _load_run_config(run_dir: Path) -> dict:
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


RUN_DIR = Path(
    "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/runs"
    "/csac_lb_temp_strict_40ep"
    "/CSACLBTemp-{CityLearnTemp-CoolingOnly-CSACLB-v0}"
    "/seed-000-2026-03-27-23-30-03"
)

# Thresholds to try: lower threshold = more cooling = more energy
thresholds = [25.0, 24.5, 24.0, 23.5, 23.0, 22.0]

print(f"{'Threshold':>10} {'Violation':>10} {'Cost KPI':>10} {'Emis KPI':>10} "
      f"{'Cons KPI':>10} {'Ramp KPI':>10} {'Mean Cool':>10}")
print("-" * 80)

# Reference: CSAC-LB final = cost 0.926, cons 0.959
print(f"{'CSAC final':>10} {'0.102':>10} {'0.926':>10} {'0.961':>10} "
      f"{'0.959':>10} {'1.074':>10} {'0.229':>10}")
print("-" * 80)

for threshold in thresholds:
    _apply_env_overrides(RUN_DIR)
    env = CityLearnTempCoolingOnlyMaskedRewardCMDP("CityLearnTemp-CoolingOnly-CSACLB-v0")
    obs, _ = env.reset(seed=0)
    terminated = truncated = False

    total_steps = active_steps = comfort_violations = 0
    action_sum = 0.0
    action_count = 0

    while not (bool(terminated) or bool(truncated)):
        actions = np.zeros(env.cooling_action_dim, dtype=np.float32)
        for i in range(env.cooling_action_dim):
            tin, tset = env._cooling_state(i)
            if tin is not None and np.isfinite(tin) and tin > threshold:
                actions[i] = 1.0

        obs, reward, cost, terminated, truncated, info = env.step(actions)
        total_steps += 1
        if float(info.get("comfort_in_warmup", 0.0)) < 0.5:
            active_steps += 1
            comfort_violations += int(float(info.get("comfort_violation", 0.0)) > 0.5)
        action_sum += float(np.sum(actions))
        action_count += int(np.size(actions))

    kpis = _evaluate_kpis(env)
    viol = comfort_violations / active_steps if active_steps > 0 else 0
    mean_cool = action_sum / action_count if action_count > 0 else 0

    print(f"{threshold:>10.1f} {viol:>10.3f} {kpis.get('cost_total', -1):>10.3f} "
          f"{kpis.get('carbon_emissions_total', -1):>10.3f} "
          f"{kpis.get('electricity_consumption_total', -1):>10.3f} "
          f"{kpis.get('ramping_average', -1):>10.3f} {mean_cool:>10.3f}")
