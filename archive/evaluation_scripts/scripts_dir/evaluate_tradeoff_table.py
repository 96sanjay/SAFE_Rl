#!/usr/bin/env python3
"""Evaluate raw and controllability-aware constraint metrics side by side.

Outputs per run:
  - C0 raw departure violations
  - C2 raw battery SoC violations
  - C3 raw building violations
  - C3 controllable violations
  - C4 raw grid violations
  - C4 avoidable violations

This is intended for thesis/report tables where raw percentages alone hide the
structural floor in C3/C4.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import yaml

PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

from omnisafe.utils.config import Config

from scripts.train_multi_lag import deep_update, load_ppolag_defaults


P_BMAX = 4.6083
P_GMAX = 10.2352


BASE_ENV = {
    "CITYLEARN_SCHEMA": f"{PROJECT}/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json",
    "CITYLEARN_CENTRAL_AGENT": "1",
    "CITYLEARN_REWARD_TYPE": "stems",
    "CITYLEARN_EXPORT_FACTOR": "0.7",
    "CITYLEARN_STEMS_P_BUILDING_MAX": str(P_BMAX),
    "CITYLEARN_STEMS_P_GRID_MAX": str(P_GMAX),
    "CITYLEARN_STEMS_SOC_LOW": "0.0",
    "CITYLEARN_STEMS_SOC_HIGH": "0.95",
    "CITYLEARN_EV_MISSING_ACTION_MODE": "assume_full",
    "CITYLEARN_INCLUDE_EV_COST": "1",
    "CITYLEARN_KPI_RUN_NAME": "__eval_tradeoff__",
    "CITYLEARN_SPATIAL_OBS": "0",
    "CITYLEARN_TEMPORAL_WINDOW": "0",
    "CITYLEARN_C3_CONTROLLABLE": "1",
    "CITYLEARN_WM_DISABLE": "1",
    "CITYLEARN_EV_ACTION_CLAMP": "0",
    "CITYLEARN_BATT_CLAMP": "0",
    "CITYLEARN_ACTION_MASK": "0",
    "CITYLEARN_POLICY_ACTION_MASK": "0",
    "CITYLEARN_COUPLED_BUDGET_ACTION": "0",
    "CITYLEARN_EV_SAUTE": "1",
    "CITYLEARN_EV_SAUTE_BUDGET": "25000",
    "CITYLEARN_EV_SAUTE_PENALTY": "5.0",
    "CITYLEARN_EV_SAUTE_GAMMA": "1.0",
    "CITYLEARN_EV_SAUTE_SHAPED_ALPHA": "10.0",
    "STEMS_ALPHA_GRID": "0.0",
    "STEMS_ALPHA_BUILD": "0.0",
    "STEMS_MU_ECONOMIC": "0.0",
    "STEMS_ALPHA_LOAD_SHIFT": "0.0",
    "STEMS_ALPHA_GRID_MILD": "0.3",
    "STEMS_ALPHA_EV_GUARD": "1.0",
    "STEMS_ALPHA_V2G_CONTEXT": "3.0",
    "STEMS_LAMBDA_EV": "5.0",
    "STEMS_ALPHA_BARRIER": "0.5",
    "STEMS_BETA_RAMP": "0.3",
    "STEMS_XI_RENEWABLE": "0.2",
    "STEMS_SB_ASYMMETRIC": "1",
    "STEMS_SG_EXPORT_CREDIT": "0.5",
    "STEMS_SG_THRESHOLD": "0.5",
    "STEMS_ALPHA_PEAK_SHAVE": "0.0",
    "STEMS_ALPHA_EV_SOLAR": "0.0",
    "STEMS_ALPHA_SOLAR_STORE": "0.0",
    "STEMS_SOLAR_STORE_BATT_ONLY": "0",
    "STEMS_EV_SLACK_ARB_SCALE": "0.0",
    "STEMS_ALPHA_HEADROOM": "0.0",
    "COST_W_C2": "0.0",
    "COST_W_C3": "5.0",
    "CITYLEARN_W_COST_EV": "1.0",
    "CITYLEARN_W_COST_SOC": "10.0",
    "CITYLEARN_W_COST_BUILDING": "0.5",
    "CITYLEARN_W_COST_GRID": "0.05",
    "CITYLEARN_EV_COST_SCALE": "3.0",
}


@dataclass
class RunSpec:
    name: str
    cfg_path: str
    checkpoint: str
    env_overrides: dict[str, str]


RUN_SPECS = [
    RunSpec(
        name="r25b_ev_slack_arb",
        cfg_path=f"{PROJECT}/configs/on-policy/r25b_ev_slack_arb.yaml",
        checkpoint=f"{PROJECT}/runs/r25b_ev_slack_arb/5bld/PPOLagMulti-{{CityLearnSafety-V2G-v2}}/seed-042-2026-03-17-17-01-55/torch_save/epoch-80.pt",
        env_overrides={"STEMS_EV_SLACK_ARB_SCALE": "2.0"},
    ),
    RunSpec(
        name="r25b_report_stable",
        cfg_path=f"{PROJECT}/configs/on-policy/r25b_report_stable.yaml",
        checkpoint=f"{PROJECT}/runs/r25b_report_stable/5bld/PPOLagMulti-{{CityLearnSafety-V2G-v2}}/seed-042-2026-03-27-04-04-26/torch_save/epoch-80.pt",
        env_overrides={"STEMS_EV_SLACK_ARB_SCALE": "2.0"},
    ),
    RunSpec(
        name="r25b_coupled_budget_10ep",
        cfg_path=f"{PROJECT}/configs/on-policy/r25b_coupled_budget_10ep.yaml",
        checkpoint=f"{PROJECT}/runs/r25b_coupled_budget_10ep/5bld/PPOLagMulti-{{CityLearnSafety-V2G-v2}}/seed-042-2026-03-27-20-34-49/torch_save/epoch-10.pt",
        env_overrides={"STEMS_EV_SLACK_ARB_SCALE": "2.0", "CITYLEARN_COUPLED_BUDGET_ACTION": "1"},
    ),
]


class MLPActor(torch.nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers += [torch.nn.Linear(in_dim, h), torch.nn.Tanh()]
            in_dim = h
        layers.append(torch.nn.Linear(in_dim, act_dim))
        self.mean = torch.nn.Sequential(*layers)

    def forward(self, obs):
        return torch.tanh(self.mean(obs))


def reset_env_vars(env_overrides: dict[str, str]) -> None:
    for k in list(os.environ.keys()):
        if k.startswith(("CITYLEARN_", "STEMS_", "COST_W_")):
            os.environ.pop(k)
    os.environ.update(BASE_ENV)
    os.environ.update(env_overrides)


def build_agent(cfg_path: str, checkpoint: str):
    with open(cfg_path) as f:
        custom = yaml.safe_load(f)
    state = torch.load(checkpoint, map_location="cpu")
    pi_state = state["pi"]
    h1 = pi_state["mean.0.weight"].shape[0]
    h2 = pi_state["mean.2.weight"].shape[0]
    obs_dim = pi_state["mean.0.weight"].shape[1]
    act_dim = pi_state["mean.4.weight"].shape[0]
    actor = MLPActor(obs_dim, act_dim, (h1, h2))
    actor.load_state_dict({k: v for k, v in pi_state.items() if not k.startswith("log_std")}, strict=False)
    actor.eval()
    obs_mean = obs_std = obs_clip = None
    if state.get("obs_normalizer") is not None:
        norm = state["obs_normalizer"]
        obs_mean = torch.as_tensor(norm["_mean"], dtype=torch.float32)
        obs_std = torch.as_tensor(norm["_std"], dtype=torch.float32)
        obs_clip = float(norm["_clip"].float().mean())
    return actor, obs_mean, obs_std, obs_clip


def build_eval_env():
    import citylearn_safe.omni_env_v2  # noqa: F401
    from citylearn_safe.omni_env_v2 import CityLearnCMDPv2

    return CityLearnCMDPv2("CityLearnSafety-V2G-v2")


def exogenous_building_power(building, t_idx: int) -> float:
    nsl = getattr(building, "_Building__energy_to_non_shiftable_load", [])
    sg = getattr(building, "_Building__solar_generation", [])
    val = 0.0
    if len(nsl) > t_idx:
        val += float(nsl[t_idx])
    if len(sg) > t_idx:
        val += float(sg[t_idx])
    return val


def max_battery_discharge_kw(building, soc: float) -> float:
    try:
        es = building.electrical_storage
        p_batt = float(getattr(es, "nominal_power", 5.0) or 5.0)
        cap = float(getattr(es, "capacity", 6.4) or 6.4)
        rte = float(getattr(es, "round_trip_efficiency", 0.9487) or 0.9487)
    except Exception:
        return 0.0
    return min(p_batt, max(0.0, soc) * cap / max(rte, 1e-6))


def battery_soc(building, t_idx: int) -> float:
    try:
        soc_arr = getattr(building.electrical_storage, "soc", None)
        if soc_arr is not None and len(soc_arr) > t_idx:
            return float(soc_arr[t_idx])
    except Exception:
        pass
    return 0.0


def ev_connected_and_max_discharge(building, t_idx: int) -> float:
    chargers = getattr(building, "electric_vehicle_chargers", None) or getattr(building, "chargers", [])
    if not chargers:
        return 0.0
    ch = chargers[0]
    sim = getattr(ch, "charger_simulation", getattr(ch, "_Charger__charger_simulation", None))
    if sim is None:
        return 0.0
    state_arr = getattr(sim, "_electric_vehicle_charger_state", None)
    if state_arr is None or t_idx >= len(state_arr) or float(state_arr[t_idx]) != 1.0:
        return 0.0
    return float(getattr(ch, "max_discharging_power", 0.0) or 0.0)


def ev_structural_safe_discharge(building, t_idx: int) -> float:
    """Upper bound on EV discharge that still respects departure feasibility."""
    chargers = getattr(building, "electric_vehicle_chargers", None) or getattr(building, "chargers", [])
    if not chargers:
        return 0.0
    ch = chargers[0]
    sim = getattr(ch, "charger_simulation", getattr(ch, "_Charger__charger_simulation", None))
    if sim is None:
        return 0.0
    state_arr = getattr(sim, "_electric_vehicle_charger_state", None)
    if state_arr is None or t_idx >= len(state_arr) or float(state_arr[t_idx]) != 1.0:
        return 0.0
    dep_arr = getattr(sim, "_electric_vehicle_departure_time", None)
    req_arr = getattr(sim, "_electric_vehicle_required_soc_departure", None)
    if dep_arr is None or req_arr is None or t_idx >= len(dep_arr) or t_idx >= len(req_arr):
        return 0.0
    t_dep = float(dep_arr[t_idx])
    if t_dep <= 2.0:
        return 0.0
    ev = getattr(ch, "connected_electric_vehicle", None)
    bt = getattr(ev, "battery", None) if ev is not None else None
    if bt is None:
        return 0.0
    soc_arr = getattr(bt, "soc", None)
    if soc_arr is None:
        return 0.0
    soc_idx = max(0, t_idx - 1)
    if soc_idx >= len(soc_arr):
        return 0.0
    soc_now = float(np.clip(float(soc_arr[soc_idx]), 0.0, 1.0))
    soc_req = float(req_arr[t_idx])
    cap_ev = float(getattr(bt, "capacity", 60.0) or 60.0)
    eta_raw = getattr(bt, "charging_efficiency", None)
    if eta_raw is not None and float(eta_raw) > 0:
        eta = float(eta_raw)
    else:
        rte = float(getattr(bt, "round_trip_efficiency", 0.9025) or 0.9025)
        eta = float(np.sqrt(max(rte, 0.01)))
    p_ev_max = float(getattr(ch, "max_charging_power", 0.0) or 0.0)
    p_ev_dis = float(getattr(ch, "max_discharging_power", 0.0) or 0.0)
    if p_ev_max <= 0.0 or p_ev_dis <= 0.0:
        return 0.0
    target_soc = min(1.0, soc_req + 0.02)
    required_input = max(0.0, target_soc - soc_now) * cap_ev / max(eta, 1e-6)
    remaining_hours = max(0.0, t_dep - 1.0)
    recoverable_input = p_ev_max * remaining_hours
    surplus_input = max(0.0, recoverable_input - required_input)
    return min(p_ev_dis, surplus_input)


def evaluate_run(spec: RunSpec) -> dict[str, float]:
    reset_env_vars(spec.env_overrides)
    actor, obs_mean, obs_std, obs_clip = build_agent(spec.cfg_path, spec.checkpoint)
    env = build_eval_env()
    obs, _ = env.reset()
    city = env._get_citylearn()
    buildings = list(city.buildings)
    n_b = len(buildings)

    steps = 0
    reward_sum = 0.0
    c0_viol = c0_total = 0
    c2_viol = c2_total = 0
    c3_raw_viol = c3_cont_viol = c3_total = 0
    c4_raw_viol = c4_avoid_viol = c4_total = 0
    ev_tracker: dict[tuple[int, int], dict[str, float | bool]] = {}

    while True:
        t_idx = int(getattr(city, "time_step", 0))
        exo_vals = [exogenous_building_power(b, t_idx) for b in buildings]
        batt_socs = [battery_soc(b, max(0, t_idx - 1)) for b in buildings]

        # Match eval_all_runs.py: detect EV departures by connection transitions.
        for b_idx, bld in enumerate(buildings):
            chargers = getattr(bld, "electric_vehicle_chargers", None) or getattr(bld, "chargers", [])
            for ch_idx, ch in enumerate(chargers):
                key = (b_idx, ch_idx)
                sim = getattr(ch, "charger_simulation", getattr(ch, "_Charger__charger_simulation", None))
                if sim is None:
                    continue
                try:
                    sa = np.asarray(getattr(sim, "_electric_vehicle_charger_state"), dtype=float)
                    connected = t_idx < len(sa) and float(sa[t_idx]) == 1.0
                    if connected:
                        ra = np.asarray(getattr(sim, "_electric_vehicle_required_soc_departure"), dtype=float)
                        rs = float(ra[t_idx]) if t_idx < len(ra) else 1.0
                        ev_obj = getattr(ch, "connected_electric_vehicle", None)
                        soc = 0.0
                        if ev_obj and getattr(ev_obj, "battery", None):
                            soc_arr = np.asarray(ev_obj.battery.soc, dtype=float)
                            soc_idx = max(0, t_idx - 1)
                            if 0 <= soc_idx < len(soc_arr):
                                soc = float(np.clip(soc_arr[soc_idx], 0.0, 1.0))
                        ev_tracker[key] = {"was": True, "soc": soc, "req": rs}
                    else:
                        prev = ev_tracker.get(key, {})
                        if prev.get("was", False):
                            c0_total += 1
                            deficit = max(0.0, float(prev["req"]) - float(prev["soc"]))
                            if deficit > 0.01:
                                c0_viol += 1
                        ev_tracker[key] = {"was": False}
                except Exception:
                    pass

        with torch.no_grad():
            obs_flat = np.asarray(obs).ravel()
            obs_t = torch.as_tensor(obs_flat, dtype=torch.float32).unsqueeze(0)
            if obs_mean is not None:
                obs_t = (obs_t - obs_mean) / (obs_std + 1e-8)
                if obs_clip is not None:
                    obs_t = obs_t.clamp(-obs_clip, obs_clip)
            act = actor(obs_t).squeeze(0).numpy()
        obs, rew, cost, terminated, truncated, info = env.step(act)
        reward_sum += float(rew.item() if hasattr(rew, "item") else rew)
        steps += 1

        c2_viol += int(info.get("battery_soc_violation_count", 0.0) or 0)
        c2_total += n_b

        c4_total += 1
        c4_raw = float(info.get("cost_stems_grid_power", 0.0)) > 0.0
        if c4_raw:
            c4_raw_viol += 1
            exo_import = sum(max(0.0, x) for x in exo_vals)
            max_dis = 0.0
            for bi, b in enumerate(buildings):
                max_dis += max_battery_discharge_kw(b, batt_socs[bi])
                max_dis += ev_structural_safe_discharge(b, t_idx)
            if (exo_import - max_dis) <= P_GMAX:
                c4_avoid_viol += 1

        for bi, b in enumerate(buildings):
            try:
                nec = getattr(b, "net_electricity_consumption", None)
                p_i = float(nec[t_idx]) if nec is not None and len(nec) > t_idx else 0.0
            except Exception:
                p_i = 0.0
            exo_i = exo_vals[bi]
            c3_total += 1
            raw_v = max(0.0, abs(p_i) - P_BMAX)
            if raw_v > 0.0:
                c3_raw_viol += 1
            if abs(exo_i) <= P_BMAX:
                cont_v = max(0.0, abs(p_i) - P_BMAX)
            else:
                cont_v = max(0.0, abs(p_i) - abs(exo_i))
            if cont_v > 0.0:
                c3_cont_viol += 1

        done = bool(terminated.item() if hasattr(terminated, "item") else terminated) or bool(
            truncated.item() if hasattr(truncated, "item") else truncated
        )
        if done:
            break

    return {
        "run": spec.name,
        "steps": steps,
        "reward_sum": reward_sum,
        "c0_pct": 100.0 * c0_viol / max(1, c0_total),
        "c2_pct": 100.0 * c2_viol / max(1, c2_total),
        "c3_raw_pct": 100.0 * c3_raw_viol / max(1, c3_total),
        "c3_controllable_pct": 100.0 * c3_cont_viol / max(1, c3_total),
        "c4_raw_pct": 100.0 * c4_raw_viol / max(1, c4_total),
        "c4_avoidable_pct": 100.0 * c4_avoid_viol / max(1, c4_total),
        "c0_viol": c0_viol,
        "c0_total": c0_total,
        "c2_viol": c2_viol,
        "c2_total": c2_total,
        "c3_raw_viol": c3_raw_viol,
        "c3_cont_viol": c3_cont_viol,
        "c3_total": c3_total,
        "c4_raw_viol": c4_raw_viol,
        "c4_avoid_viol": c4_avoid_viol,
        "c4_total": c4_total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=f"{PROJECT}/runs/final_tradeoff_table.csv")
    args = parser.parse_args()

    rows = []
    for spec in RUN_SPECS:
        print(f"Evaluating {spec.name} ...")
        rows.append(evaluate_run(spec))

    fields = [
        "run",
        "c0_pct",
        "c2_pct",
        "c3_raw_pct",
        "c3_controllable_pct",
        "c4_raw_pct",
        "c4_avoidable_pct",
        "reward_sum",
    ]
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})

    print("\nFinal table:")
    for row in rows:
        print(
            f"{row['run']}: "
            f"C0={row['c0_pct']:.2f}%  "
            f"C2={row['c2_pct']:.2f}%  "
            f"C3raw={row['c3_raw_pct']:.2f}%  "
            f"C3ctrl={row['c3_controllable_pct']:.2f}%  "
            f"C4raw={row['c4_raw_pct']:.2f}%  "
            f"C4avoid={row['c4_avoidable_pct']:.2f}%"
        )
    print(f"\nSaved CSV: {args.csv}")


if __name__ == "__main__":
    main()
