#!/usr/bin/env python3
"""
Full KPI Evaluation: R5a (MLP) vs PPOLagMulti (Softmax).

Computes per-constraint violation percentages, cost totals, CityLearn KPIs,
and detailed comparison tables. Determinism verified via double-run.

Each agent uses its own env profile (R5a: 198-dim, PPOLagMulti: 218-dim).
"""
import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn

PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

EVAL_SEED = 42
random.seed(EVAL_SEED)
np.random.seed(EVAL_SEED)
torch.manual_seed(EVAL_SEED)

# =========================================================================
# Environment profiles
# =========================================================================
COMMON_ENV_VARS = {
    "CITYLEARN_SCHEMA": f"{PROJECT}/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json",
    "CITYLEARN_CENTRAL_AGENT": "1",
    "CITYLEARN_REWARD_TYPE": "stems",
    "CITYLEARN_EXPORT_FACTOR": "0.7",
    "CITYLEARN_STEMS_P_BUILDING_MAX": "4.6083",
    "CITYLEARN_STEMS_P_GRID_MAX": "10.2352",
    "CITYLEARN_STEMS_SOC_LOW": "0.0",
    "CITYLEARN_STEMS_SOC_HIGH": "0.95",
    "CITYLEARN_STEMS_PNORM_P": "4.0",
    "CITYLEARN_W_COST_EV": "1.0",
    "CITYLEARN_W_COST_SOC": "10.0",
    "CITYLEARN_W_COST_BUILDING": "0.5",
    "CITYLEARN_W_COST_GRID": "0.05",
    "CITYLEARN_EV_COST_SCALE": "3.0",
    "CITYLEARN_EV_MISSING_ACTION_MODE": "assume_full",
    "CITYLEARN_INCLUDE_EV_COST": "1",
    "CITYLEARN_KPI_RUN_NAME": "__eval_disabled__",
}

PROFILE_198 = {
    **COMMON_ENV_VARS,
    "CITYLEARN_C3_CONTROLLABLE": "0",
    "CITYLEARN_SPATIAL_OBS": "0",
}

PROFILE_218 = {
    **COMMON_ENV_VARS,
    "CITYLEARN_C3_CONTROLLABLE": "1",
    "CITYLEARN_SPATIAL_OBS": "1",
    "COST_W_C1": "10.0",
    "COST_W_C1_DENSE": "5.0",
    "COST_W_C2": "1.0",
    "COST_W_C3": "0.1",
    "COST_W_C4": "5.0",
    "CITYLEARN_EV_DENSE_COST_SCALE": "1.0",
    "STEMS_ALPHA_GRID": "1.0",
    "STEMS_BETA_RAMP": "2.0",
}


def set_env_profile(profile: dict):
    stale_keys = [
        "CITYLEARN_C3_CONTROLLABLE", "CITYLEARN_SPATIAL_OBS",
        "COST_W_C1", "COST_W_C1_DENSE", "COST_W_C2", "COST_W_C3", "COST_W_C4",
        "CITYLEARN_EV_DENSE_COST_SCALE", "STEMS_ALPHA_GRID", "STEMS_BETA_RAMP",
    ]
    for k in stale_keys:
        os.environ.pop(k, None)
    for k, v in profile.items():
        os.environ[k] = v


# Initial profile
set_env_profile(PROFILE_198)

from scripts.make_env import make_base_env
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
from citylearn_safe.stems_obs_wrapper import SpatialGraphFeaturesWrapper
from citylearn_safe.extractors_v3 import unwrap_to_raw_citylearn_env
import citylearn_safe.schema_index as si


class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        self.mean = nn.Sequential(*layers)

    def forward(self, obs):
        return torch.tanh(self.mean(obs))


def load_actor(ckpt_path, obs_dim, act_dim):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pi_state = ckpt["pi"]

    h1 = pi_state["mean.0.weight"].shape[0]
    h2 = pi_state["mean.2.weight"].shape[0]
    actual_obs_dim = pi_state["mean.0.weight"].shape[1]
    actual_act_dim = pi_state["mean.4.weight"].shape[0]

    if actual_obs_dim != obs_dim:
        print(f"  WARNING: expected obs_dim={obs_dim}, ckpt has {actual_obs_dim}. Using ckpt value.")
        obs_dim = actual_obs_dim
    if actual_act_dim != act_dim:
        print(f"  WARNING: expected act_dim={act_dim}, ckpt has {actual_act_dim}. Using ckpt value.")
        act_dim = actual_act_dim

    actor = MLPActor(obs_dim, act_dim, (h1, h2))
    filtered = {k: v for k, v in pi_state.items() if not k.startswith("log_std")}
    result = actor.load_state_dict(filtered, strict=False)
    if result.missing_keys:
        raise RuntimeError(f"Missing keys: {result.missing_keys}")
    actor.eval()

    obs_mean = obs_std = obs_clip = None
    if "obs_normalizer" in ckpt:
        norm = ckpt["obs_normalizer"]
        obs_mean = torch.as_tensor(norm["_mean"], dtype=torch.float32)
        obs_std = torch.as_tensor(norm["_std"], dtype=torch.float32)
        clip_t = norm["_clip"].float()
        obs_clip = float(clip_t.mean())

    return actor, obs_mean, obs_std, obs_clip, obs_dim, act_dim


def run_episode(actor, obs_mean, obs_std, obs_clip, label, use_spatial=False, seed=42):
    """Run 1 deterministic episode. Returns dict of all metrics."""
    si._CACHE = None
    base = make_base_env(central_agent=True)
    safety = CityLearnSafetyEnvV3(base)
    env = ForecastObsWrapper(safety, forecast_horizon=24)

    if use_spatial:
        p_bmax = float(os.environ.get("CITYLEARN_STEMS_P_BUILDING_MAX", "4.6083"))
        env = SpatialGraphFeaturesWrapper(env, num_buildings=5, p_building_max=p_bmax)

    # Get raw CityLearn env for evaluate()
    city = base
    for _ in range(20):
        if hasattr(city, 'buildings') and len(getattr(city, 'buildings', [])) > 0:
            break
        city = getattr(city, 'env', getattr(city, 'base', getattr(city, 'unwrapped', None)))

    raw = unwrap_to_raw_citylearn_env(env)
    buildings = list(getattr(raw, "buildings", []))

    # Action name parsing
    names_raw = getattr(raw, "action_names", [])
    if isinstance(names_raw, list) and len(names_raw) == 1 and isinstance(names_raw[0], list):
        names = names_raw[0]
    else:
        names = list(names_raw)
    batt_idx = [i for i, n in enumerate(names) if str(n).strip().lower() == "electrical_storage"]
    ev_idx = [i for i, n in enumerate(names) if "electric_vehicle_storage_charger_" in str(n).lower()]

    obs, _ = env.reset(seed=seed)
    done = False
    step = 0

    rewards = []
    costs = []
    actions_all = []

    c1_vals, c2_vals, c3_vals, c4_vals = [], [], [], []
    c0_vals = []  # C0: EV dense cost (from info)
    c1_steps = c2_steps = c3_steps = c4_steps = c0_steps = 0

    # Per-hour tracking for diurnal profiles
    hourly_actions = {h: [] for h in range(24)}
    hourly_batt = {h: [] for h in range(24)}
    hourly_ev = {h: [] for h in range(24)}

    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        if obs_mean is not None:
            obs_t = (obs_t - obs_mean) / (obs_std + 1e-8)
            if obs_clip is not None:
                obs_t = obs_t.clamp(-obs_clip, obs_clip)
        with torch.no_grad():
            action_t = actor(obs_t).squeeze(0).numpy()
        action = np.clip(action_t, -1.0, 1.0)
        actions_all.append(action.copy())

        t_now = int(getattr(raw, "time_step", 0))
        hour = t_now % 24
        hourly_actions[hour].append(action.copy())
        if batt_idx:
            hourly_batt[hour].append(np.mean([action[i] for i in batt_idx]))
        if ev_idx:
            hourly_ev[hour].append(np.mean([action[i] for i in ev_idx]))

        obs, reward, term, trunc, info = env.step(action)
        done = bool(term) or bool(trunc)
        step += 1

        r = float(reward)
        c = float(info.get("cost", 0.0))
        rewards.append(r)
        costs.append(c)

        c0 = float(info.get("cost_ev_dense", 0.0))
        c1 = float(info.get("cost_ev_departure", 0.0))
        c2 = float(info.get("cost_stems_battery", 0.0))
        c3 = float(info.get("cost_stems_building_power", 0.0))
        c4 = float(info.get("cost_stems_grid_power", 0.0))

        c0_vals.append(c0); c1_vals.append(c1); c2_vals.append(c2)
        c3_vals.append(c3); c4_vals.append(c4)

        if c0 > 0: c0_steps += 1
        if c1 > 0: c1_steps += 1
        if c2 > 0: c2_steps += 1
        if c3 > 0: c3_steps += 1
        if c4 > 0: c4_steps += 1

    # CityLearn evaluate()
    cl_kpis = {}
    try:
        if city and hasattr(city, 'evaluate'):
            import pandas as pd
            eval_df = city.evaluate()
            if eval_df is not None and not eval_df.empty:
                if "name" in eval_df.columns:
                    district = eval_df[eval_df["name"] == "District"]
                else:
                    district = eval_df
                for _, row in district.iterrows():
                    cf = str(row.get("cost_function", ""))
                    val = row.get("value", None)
                    if cf and val is not None:
                        try:
                            cl_kpis[cf] = float(val)
                        except (ValueError, TypeError):
                            pass
    except Exception as e:
        print(f"  [{label}] CityLearn evaluate() error: {e}")

    actions_arr = np.array(actions_all)
    total_steps = step

    # Hourly profiles
    hourly_batt_means = np.array([np.mean(hourly_batt[h]) if hourly_batt[h] else 0 for h in range(24)])
    hourly_ev_means = np.array([np.mean(hourly_ev[h]) if hourly_ev[h] else 0 for h in range(24)])

    results = {
        "label": label,
        "total_steps": total_steps,

        # Reward
        "total_reward": sum(rewards),
        "avg_reward": sum(rewards) / total_steps,

        # Total CMDP cost
        "total_cost": sum(costs),
        "avg_cost": sum(costs) / total_steps,
        "cost_positive_steps": sum(1 for c in costs if c > 0),
        "cost_positive_pct": 100.0 * sum(1 for c in costs if c > 0) / total_steps,

        # Cost component totals
        "c0_ev_dense_total": sum(c0_vals),
        "c1_ev_total": sum(c1_vals),
        "c2_soc_total": sum(c2_vals),
        "c3_building_total": sum(c3_vals),
        "c4_grid_total": sum(c4_vals),

        # Violation percentages
        "c0_ev_dense_violation_pct": 100.0 * c0_steps / total_steps,
        "c1_ev_violation_pct": 100.0 * c1_steps / total_steps,
        "c2_soc_violation_pct": 100.0 * c2_steps / total_steps,
        "c3_building_violation_pct": 100.0 * c3_steps / total_steps,
        "c4_grid_violation_pct": 100.0 * c4_steps / total_steps,

        # Cost component shares
        "c1_ev_share_pct": 100.0 * sum(c1_vals) / max(sum(costs), 1e-8),
        "c2_soc_share_pct": 100.0 * sum(c2_vals) / max(sum(costs), 1e-8),
        "c3_building_share_pct": 100.0 * sum(c3_vals) / max(sum(costs), 1e-8),
        "c4_grid_share_pct": 100.0 * sum(c4_vals) / max(sum(costs), 1e-8),

        # Action statistics
        "action_mean": float(actions_arr.mean()),
        "action_std": float(actions_arr.std()),
        "action_abs_mean": float(np.abs(actions_arr).mean()),

        # Battery/EV specific
        "batt_mean": float(np.mean([actions_arr[:, i].mean() for i in batt_idx])) if batt_idx else 0,
        "ev_mean": float(np.mean([actions_arr[:, i].mean() for i in ev_idx])) if ev_idx else 0,

        # Hourly profiles (for analysis)
        "hourly_batt": hourly_batt_means.tolist(),
        "hourly_ev": hourly_ev_means.tolist(),

        # CityLearn KPIs
        **{f"cl_{k}": v for k, v in cl_kpis.items()},

        # Hash for determinism
        "_reward_hash": float(np.array(rewards).sum()),
        "_cost_hash": float(np.array(costs).sum()),
        "_action_hash": float(actions_arr.sum()),
    }

    env.close()
    return results


def main():
    OUT_DIR = f"{PROJECT}/runs/multi_lag/5bld/evaluation"
    os.makedirs(OUT_DIR, exist_ok=True)

    AGENTS = {
        "R5a (MLP)": {
            "ckpt": f"{PROJECT}/runs/r6_compare/r5a_5bld/PPOLag-{{CityLearnSafety-V2G-v2}}/seed-000-2026-03-06-11-04-24/torch_save/epoch-50.pt",
            "obs_dim": 198,
            "profile": PROFILE_198,
            "use_spatial": False,
        },
        "PPOLagMulti (Softmax)": {
            "ckpt": f"{PROJECT}/runs/multi_lag/5bld/PPOLagMulti-{{CityLearnSafety-V2G-v2}}/seed-042-2026-03-08-22-54-20/torch_save/epoch-50.pt",
            "obs_dim": 218,
            "profile": PROFILE_218,
            "use_spatial": True,
        },
    }

    all_results = {}

    for agent_name, agent_cfg in AGENTS.items():
        print(f"\n{'='*60}")
        print(f"  EVALUATING: {agent_name}")
        print(f"  Checkpoint: {agent_cfg['ckpt']}")
        print(f"{'='*60}")

        set_env_profile(agent_cfg["profile"])

        actor, obs_mean, obs_std, obs_clip, actual_obs_dim, act_dim = load_actor(
            agent_cfg["ckpt"], agent_cfg["obs_dim"], 9
        )
        print(f"  Loaded: obs_dim={actual_obs_dim}, act_dim={act_dim}")

        random.seed(EVAL_SEED)
        np.random.seed(EVAL_SEED)
        torch.manual_seed(EVAL_SEED)

        print(f"  Run 1 (primary)...")
        r1 = run_episode(actor, obs_mean, obs_std, obs_clip, agent_name,
                         use_spatial=agent_cfg["use_spatial"], seed=EVAL_SEED)

        print(f"  Run 2 (determinism check)...")
        random.seed(EVAL_SEED)
        np.random.seed(EVAL_SEED)
        torch.manual_seed(EVAL_SEED)
        r2 = run_episode(actor, obs_mean, obs_std, obs_clip, agent_name,
                         use_spatial=agent_cfg["use_spatial"], seed=EVAL_SEED)

        reward_match = abs(r1["_reward_hash"] - r2["_reward_hash"]) < 1e-4
        cost_match = abs(r1["_cost_hash"] - r2["_cost_hash"]) < 1e-4
        if reward_match and cost_match:
            print(f"  DETERMINISM VERIFIED")
        else:
            print(f"  WARNING: RESULTS DIFFER!")
            print(f"    Reward: {r1['_reward_hash']:.6f} vs {r2['_reward_hash']:.6f}")
            print(f"    Cost:   {r1['_cost_hash']:.6f} vs {r2['_cost_hash']:.6f}")

        all_results[agent_name] = r1

    # =====================================================================
    # REPORT
    # =====================================================================
    report = []
    def p(line=""):
        print(line)
        report.append(line)

    agents = list(all_results.keys())

    p()
    p("=" * 100)
    p("  FULL KPI EVALUATION: R5a (MLP) vs PPOLagMulti (Softmax)")
    p("  Environment: 5-building CityLearn V2G, seed=42, deterministic")
    p("=" * 100)

    header = f"  {'Metric':<45}" + "".join(f"{a.split('(')[0].strip():>20}" for a in agents) + f"{'Delta':>14}"
    p()
    p(header)
    p("  " + "-" * (len(header) - 2))

    def row(metric, key, fmt=".1f", lower_better=None):
        vals = [all_results[a].get(key, float('nan')) for a in agents]
        strs = []
        for v in vals:
            if isinstance(v, float) and (np.isnan(v) or v == 0 and key.startswith("cl_")):
                strs.append("N/A")
            else:
                strs.append(f"{v:{fmt}}")

        # Delta
        if len(vals) == 2 and not any(isinstance(v, float) and np.isnan(v) for v in vals):
            delta = vals[1] - vals[0]
            if lower_better is not None:
                better = (lower_better and delta < 0) or (not lower_better and delta > 0)
                marker = " +" if better else " -"
            else:
                marker = ""
            delta_str = f"{delta:+{fmt}}{marker}"
        else:
            delta_str = ""

        line = f"  {metric:<45}" + "".join(f"{s:>20}" for s in strs) + f"{delta_str:>14}"
        p(line)

    p()
    p("  --- REWARD ---")
    row("Total Reward (STEMS)", "total_reward", ".0f", True)
    row("Avg Reward/Step", "avg_reward", ".4f", True)

    p()
    p("  --- TOTAL CMDP COST ---")
    row("Total Cost (weighted sum)", "total_cost", ".0f", False)
    row("Avg Cost/Step", "avg_cost", ".4f", False)
    row("Steps with Any Violation (%)", "cost_positive_pct", ".1f", False)

    p()
    p("  --- PER-CONSTRAINT COST TOTALS ---")
    row("C0: EV Dense Charging", "c0_ev_dense_total", ".1f", False)
    row("C1: EV Departure", "c1_ev_total", ".1f", False)
    row("C2: Battery SoC", "c2_soc_total", ".1f", False)
    row("C3: Building Power", "c3_building_total", ".1f", False)
    row("C4: Grid Power", "c4_grid_total", ".1f", False)

    p()
    p("  --- VIOLATION RATES (% of steps with violation > 0) ---")
    row("C0: EV Dense Violation %", "c0_ev_dense_violation_pct", ".2f", False)
    row("C1: EV Departure Violation %", "c1_ev_violation_pct", ".2f", False)
    row("C2: Battery SoC Violation %", "c2_soc_violation_pct", ".2f", False)
    row("C3: Building Power Violation %", "c3_building_violation_pct", ".2f", False)
    row("C4: Grid Power Violation %", "c4_grid_violation_pct", ".2f", False)

    p()
    p("  --- COST COMPONENT SHARES (% of total CMDP cost) ---")
    row("C1 Share %", "c1_ev_share_pct", ".1f")
    row("C2 Share %", "c2_soc_share_pct", ".1f")
    row("C3 Share %", "c3_building_share_pct", ".1f")
    row("C4 Share %", "c4_grid_share_pct", ".1f")

    # Weighted cost breakdown
    p()
    p("  --- WEIGHTED COST CONTRIBUTIONS (weights: C1*1.0, C2*10.0, C3*0.5, C4*0.05) ---")
    w_ev, w_soc, w_bld, w_grid = 1.0, 10.0, 0.5, 0.05
    for name_a in agents:
        r = all_results[name_a]
        wc1 = r["c1_ev_total"] * w_ev
        wc2 = r["c2_soc_total"] * w_soc
        wc3 = r["c3_building_total"] * w_bld
        wc4 = r["c4_grid_total"] * w_grid
        wsum = wc1 + wc2 + wc3 + wc4
        short = name_a.split("(")[0].strip()
        p(f"    {short}: C1={wc1:.0f} + C2={wc2:.0f} + C3={wc3:.0f} + C4={wc4:.0f} = {wsum:.0f}")

    p()
    p("  --- CITYLEARN STANDARD KPIs (from evaluate(), 1.0 = no-op baseline) ---")
    for cl_key, cl_label in [
        ("cl_electricity_consumption_total", "Electricity Consumption"),
        ("cl_carbon_emissions_total", "Carbon Emissions"),
        ("cl_cost_total", "Electricity Cost ($)"),
        ("cl_daily_peak_average", "Daily Peak Average"),
        ("cl_all_time_peak_average", "All-Time Peak Average"),
        ("cl_ramping_average", "Ramping Average"),
        ("cl_1 - Loss of Life Share", "1 - Loss of Life Share"),
        ("cl_zero_net_energy", "Zero Net Energy"),
    ]:
        row(cl_label, cl_key, ".4f")

    p()
    p("  --- ACTION STATISTICS ---")
    row("Action Mean (all)", "action_mean", ".4f")
    row("Action Std (all)", "action_std", ".4f")
    row("Action |Mean| (all)", "action_abs_mean", ".4f")
    row("Battery Mean Action", "batt_mean", ".4f")
    row("EV Mean Action", "ev_mean", ".4f")

    # Summary comparison
    p()
    p("  " + "=" * 96)
    p("  SUMMARY: PPOLagMulti vs R5a")
    p("  " + "=" * 96)

    r5a = all_results.get("R5a (MLP)", {})
    multi = all_results.get("PPOLagMulti (Softmax)", {})
    if r5a and multi:
        comparisons = [
            ("Total Reward", "total_reward", False),
            ("Total Cost", "total_cost", True),
            ("C0 EV Dense Viol %", "c0_ev_dense_violation_pct", True),
            ("C1 EV Depart Viol %", "c1_ev_violation_pct", True),
            ("C2 SoC Viol %", "c2_soc_violation_pct", True),
            ("C3 Building Viol %", "c3_building_violation_pct", True),
            ("C4 Grid Viol %", "c4_grid_violation_pct", True),
            ("C1 Total Cost", "c1_ev_total", True),
            ("C2 Total Cost", "c2_soc_total", True),
            ("C3 Total Cost", "c3_building_total", True),
            ("C4 Total Cost", "c4_grid_total", True),
        ]
        for metric, key, lower_better in comparisons:
            v5 = r5a.get(key, 0)
            vm = multi.get(key, 0)
            if abs(v5) > 1e-6:
                pct = 100 * (vm - v5) / abs(v5)
                if lower_better:
                    direction = "BETTER" if pct < 0 else "WORSE"
                else:
                    direction = "BETTER" if pct > 0 else "WORSE"
                p(f"    {metric:<30} R5a={v5:>12.1f}  Multi={vm:>12.1f}  {pct:+7.1f}% ({direction})")
            else:
                p(f"    {metric:<30} R5a={v5:>12.1f}  Multi={vm:>12.1f}")

    p()
    p("=" * 100)

    # Save report
    with open(f"{OUT_DIR}/kpi_report.txt", "w") as f:
        f.write("\n".join(report))
    print(f"\nReport: {OUT_DIR}/kpi_report.txt")

    # Save JSON (without hourly profiles for cleanliness)
    json_results = {}
    for name in agents:
        r = dict(all_results[name])
        r.pop("hourly_batt", None)
        r.pop("hourly_ev", None)
        json_results[name] = r
    with open(f"{OUT_DIR}/kpi_results.json", "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"JSON:   {OUT_DIR}/kpi_results.json")


if __name__ == "__main__":
    main()
