#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np
import torch

PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)


def set_training_env() -> None:
    os.environ.update(
        {
            "CITYLEARN_SCHEMA": f"{PROJECT}/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json",
            "CITYLEARN_CENTRAL_AGENT": "1",
            "CITYLEARN_REWARD_TYPE": "stems",
            "CITYLEARN_EXPORT_FACTOR": "0.7",
            "CITYLEARN_STEMS_P_BUILDING_MAX": "4.6083",
            "CITYLEARN_STEMS_P_GRID_MAX": "10.2352",
            "CITYLEARN_STEMS_SOC_LOW": "0.0",
            "CITYLEARN_STEMS_SOC_HIGH": "0.95",
            "CITYLEARN_STEMS_PNORM_P": "4.0",
            "CITYLEARN_ACTION_MASK": "0",
            "CITYLEARN_BETA_ACTOR": "0",
            "CITYLEARN_EV_SAUTE": "1",
            "CITYLEARN_EV_SAUTE_BUDGET": "25000",
            "CITYLEARN_EV_SAUTE_PENALTY": "5.0",
            "CITYLEARN_EV_SAUTE_GAMMA": "1.0",
            "CITYLEARN_EV_SAUTE_SHAPED_ALPHA": "10.0",
            "CITYLEARN_PID_LAGRANGE": "1",
            "STEMS_ALPHA_GRID": "0.0",
            "STEMS_SG_THRESHOLD": "0.5",
            "STEMS_ALPHA_LOAD_SHIFT": "0.0",
            "STEMS_ALPHA_GRID_MILD": "0.3",
            "STEMS_MU_ECONOMIC": "0.0",
            "STEMS_ALPHA_BUILD": "0.0",
            "STEMS_XI_RENEWABLE": "0.2",
            "STEMS_BETA_RAMP": "0.3",
            "STEMS_LAMBDA_EV": "5.0",
            "STEMS_SB_ASYMMETRIC": "1",
            "STEMS_SG_EXPORT_CREDIT": "0.5",
            "STEMS_ALPHA_BARRIER": "0.5",
            "STEMS_ALPHA_PEAK_SHAVE": "0.0",
            "STEMS_ALPHA_EV_GUARD": "1.0",
            "STEMS_ALPHA_V2G_CONTEXT": "3.0",
            "STEMS_ALPHA_EV_SOLAR": "0.0",
            "STEMS_ALPHA_SOLAR_STORE": "0.0",
            "STEMS_SOLAR_STORE_BATT_ONLY": "0",
            "STEMS_ALPHA_HEADROOM": "0.0",
            "STEMS_ALPHA_PRICE_ARB": "0.0",
            "STEMS_ALPHA_GRID_PENALTY": "0.0",
            "STEMS_ALPHA_NEC_SIGN": "0.0",
            "STEMS_EV_SLACK_ARB_SCALE": "0.0",
            "COST_W_C2": "0.0",
            "COST_W_C3": "5.0",
            "CITYLEARN_W_COST_EV": "1.0",
            "CITYLEARN_W_COST_SOC": "10.0",
            "CITYLEARN_W_COST_BUILDING": "0.5",
            "CITYLEARN_W_COST_GRID": "0.05",
            "CITYLEARN_EV_COST_SCALE": "3.0",
            "CITYLEARN_EV_MISSING_ACTION_MODE": "assume_full",
            "CITYLEARN_INCLUDE_EV_COST": "1",
            "CITYLEARN_C3_CONTROLLABLE": "1",
            "CITYLEARN_WM_DISABLE": "1",
            "CITYLEARN_EV_ACTION_CLAMP": "0",
            "CITYLEARN_BATT_CLAMP": "0",
            "CITYLEARN_SPATIAL_OBS": "0",
            "CITYLEARN_TEMPORAL_WINDOW": "0",
        }
    )


import citylearn_safe.omni_env_v2  # noqa: E402,F401
from citylearn_safe.diff_projector import DiffProjector  # noqa: E402
from citylearn_safe.omni_env_v2 import CityLearnCMDPv2  # noqa: E402

P_BMAX = 4.6083
P_GMAX = 10.2352


class MLPActor(torch.nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        layers: list[torch.nn.Module] = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers += [torch.nn.Linear(in_dim, h), torch.nn.Tanh()]
            in_dim = h
        layers.append(torch.nn.Linear(in_dim, act_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))


def load_actor(checkpoint: str) -> MLPActor:
    state = torch.load(checkpoint, map_location="cpu")
    pi_state = state["pi"]
    h1 = pi_state["net.0.weight"].shape[0]
    h2 = pi_state["net.2.weight"].shape[0]
    obs_dim = pi_state["net.0.weight"].shape[1]
    act_dim = pi_state["net.4.weight"].shape[0]
    actor = MLPActor(obs_dim, act_dim, (h1, h2))
    actor.load_state_dict(pi_state)
    actor.eval()
    return actor


def exogenous_building_power(building, t_idx: int) -> float:
    nsl = getattr(building, "_Building__energy_to_non_shiftable_load", [])
    sg = getattr(building, "_Building__solar_generation", [])
    val = 0.0
    if len(nsl) > t_idx:
        val += float(nsl[t_idx])
    if len(sg) > t_idx:
        val += float(sg[t_idx])
    return val


def battery_soc(building, t_idx: int) -> float:
    try:
        soc_arr = getattr(building.electrical_storage, "soc", None)
        if soc_arr is not None and len(soc_arr) > t_idx:
            return float(soc_arr[t_idx])
    except Exception:
        pass
    return 0.0


def max_battery_discharge_kw(building, soc: float) -> float:
    try:
        es = building.electrical_storage
        p_batt = float(getattr(es, "nominal_power", 5.0) or 5.0)
        cap = float(getattr(es, "capacity", 6.4) or 6.4)
        rte = float(getattr(es, "round_trip_efficiency", 0.9487) or 0.9487)
    except Exception:
        return 0.0
    return min(p_batt, max(0.0, soc) * cap / max(rte, 1e-6))


def ev_structural_safe_discharge(building, t_idx: int) -> float:
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


@dataclass
class EvalResult:
    mode: str
    steps: int
    reward_sum: float
    c0_viol: int
    c0_total: int
    c0_pct: float
    c1_viol: int
    c1_total: int
    c1_pct: float
    c2_viol: int
    c2_total: int
    c2_pct: float
    c3_raw_viol: int
    c3_total: int
    c3_raw_pct: float
    c3_cont_viol: int
    c3_controllable_pct: float
    c4_raw_viol: int
    c4_total: int
    c4_raw_pct: float
    c4_avoid_viol: int
    c4_avoidable_pct: float
    proj_changed_steps: int
    proj_change_pct: float
    proj_mean_delta: float


def evaluate(checkpoint: str, projected: bool) -> EvalResult:
    set_training_env()
    actor = load_actor(checkpoint)
    env = CityLearnCMDPv2("CityLearnSafety-V2G-v2")
    obs, _ = env.reset()
    city = env._get_citylearn()
    buildings = list(city.buildings)
    n_b = len(buildings)
    projector = None
    if projected:
        projector = DiffProjector(env=env, solver_eps=1e-4, solver_max_iters=5000)
        projector.build()

    steps = 0
    reward_sum = 0.0
    proj_changed_steps = 0
    proj_delta_sum = 0.0
    c0_viol = c0_total = 0
    c1_viol = c1_total = 0
    c2_viol = c2_total = 0
    c3_raw_viol = c3_cont_viol = c3_total = 0
    c4_raw_viol = c4_avoid_viol = c4_total = 0
    ev_tracker: dict[tuple[int, int], dict[str, float | bool]] = {}

    while True:
        t_idx = int(getattr(city, "time_step", 0))
        exo_vals = [exogenous_building_power(b, t_idx) for b in buildings]
        batt_socs = [battery_soc(b, max(0, t_idx - 1)) for b in buildings]

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
                        req = float(ra[t_idx]) if t_idx < len(ra) else 1.0
                        ev_obj = getattr(ch, "connected_electric_vehicle", None)
                        soc = 0.0
                        if ev_obj and getattr(ev_obj, "battery", None):
                            soc_arr = np.asarray(ev_obj.battery.soc, dtype=float)
                            soc_idx = max(0, t_idx - 1)
                            if 0 <= soc_idx < len(soc_arr):
                                soc = float(np.clip(soc_arr[soc_idx], 0.0, 1.0))
                        ev_tracker[key] = {"was": True, "soc": soc, "req": req}
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
            obs_t = torch.as_tensor(np.asarray(obs).ravel(), dtype=torch.float32).unsqueeze(0)
            act = actor(obs_t)
            if projector is not None:
                act_safe, _ = projector.project(obs_t, act)
                delta = float(torch.norm(act_safe - act, dim=-1).mean().item())
                proj_delta_sum += delta
                if delta > 1e-6:
                    proj_changed_steps += 1
                act_np = act_safe.squeeze(0).cpu().numpy()
            else:
                act_np = act.squeeze(0).cpu().numpy()

        obs, rew, _, terminated, truncated, info = env.step(act_np)
        reward_sum += float(rew.item() if hasattr(rew, "item") else rew)
        steps += 1

        c1_total += 1
        if float(info.get("cost_ev_dense", 0.0) or 0.0) > 0.0:
            c1_viol += 1

        c2_viol += int(info.get("battery_soc_violation_count", 0.0) or 0)
        c2_total += n_b

        c4_total += 1
        c4_raw = float(info.get("cost_stems_grid_power", 0.0) or 0.0) > 0.0
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

    return EvalResult(
        mode="projected" if projected else "raw",
        steps=steps,
        reward_sum=reward_sum,
        c0_viol=c0_viol,
        c0_total=c0_total,
        c0_pct=100.0 * c0_viol / max(1, c0_total),
        c1_viol=c1_viol,
        c1_total=c1_total,
        c1_pct=100.0 * c1_viol / max(1, c1_total),
        c2_viol=c2_viol,
        c2_total=c2_total,
        c2_pct=100.0 * c2_viol / max(1, c2_total),
        c3_raw_viol=c3_raw_viol,
        c3_total=c3_total,
        c3_raw_pct=100.0 * c3_raw_viol / max(1, c3_total),
        c3_cont_viol=c3_cont_viol,
        c3_controllable_pct=100.0 * c3_cont_viol / max(1, c3_total),
        c4_raw_viol=c4_raw_viol,
        c4_total=c4_total,
        c4_raw_pct=100.0 * c4_raw_viol / max(1, c4_total),
        c4_avoid_viol=c4_avoid_viol,
        c4_avoidable_pct=100.0 * c4_avoid_viol / max(1, c4_total),
        proj_changed_steps=proj_changed_steps,
        proj_change_pct=100.0 * proj_changed_steps / max(1, steps),
        proj_mean_delta=proj_delta_sum / max(1, steps),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=["raw", "projected", "both"], default="both")
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    modes = ["raw", "projected"] if args.mode == "both" else [args.mode]
    results = {mode: asdict(evaluate(args.checkpoint, projected=(mode == "projected"))) for mode in modes}
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
