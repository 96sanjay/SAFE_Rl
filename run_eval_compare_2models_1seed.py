#!/usr/bin/env python3
import os, sys, time
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
sys.path.insert(0, REPO_ROOT)

os.environ["PYTHONPATH"] = f"{REPO_ROOT}:" + os.environ.get("PYTHONPATH", "")
os.environ["CITYLEARN_SCHEMA"] = f"{REPO_ROOT}/data/citylearn_challenge_2022_phase_all_plus_evs/schema.json"
os.environ["CITYLEARN_EXPORT_FACTOR"] = "0.7"
os.environ["CITYLEARN_REWARD_SCALE"] = "1.0"
os.environ["CITYLEARN_EV_COST_SCALE"] = "3.0"
os.environ["CITYLEARN_KPI_FLUSH_EVERY_STEP"] = "0"

from scripts.make_env import make_base_env
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.extractors_v3 import run_policy_and_oracle_rollouts, summarize_oracle_gap

import evaluate_all_policies as E

SEED = 42
MODELS = ["RBC-Greedy", "PPO-Lag-Lambda40"]

def make_env_factory():
    def _make():
        base = make_base_env(central_agent=True)
        return CityLearnSafetyEnvV3(base)
    return _make

def pick_configs():
    cfgs = []
    for c in E.MODEL_REGISTRY:
        if c.name in MODELS:
            cfgs.append(c)
    missing = [m for m in MODELS if m not in [c.name for c in cfgs]]
    if missing:
        raise SystemExit(f"Missing configs in MODEL_REGISTRY: {missing}")
    return cfgs

def to_float(x):
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")

def get_attr(obj, name, default=np.nan):
    return to_float(getattr(obj, name, default))

def main():
    out_dir = Path("runs/eval_compare_2models_1seed_FIXED_RBC")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    cfgs = pick_configs()

    for cfg in cfgs:
        print("\n" + "="*90)
        print(f"[{cfg.name}] seed={SEED} running policy+oracle...")
        print("="*90)

        t0 = time.time()
        policy_fn = E.load_policy(cfg)

        policy_summary, oracle_summary = run_policy_and_oracle_rollouts(
            make_env=make_env_factory(),
            policy_action_fn=policy_fn,
            seed=SEED
        )
        gap = summarize_oracle_gap(policy_summary, oracle_summary)
        elapsed = time.time() - t0

        r = {
            "model_name": cfg.name,
            "seed": SEED,
            "steps": int(get_attr(policy_summary, "steps", 0) or 0),

            # summed safety costs
            "sum_cost": get_attr(policy_summary, "cost_total", np.nan),
            "sum_cost_ev_departure": get_attr(policy_summary, "cost_ev_departure", np.nan),
            "sum_cost_grid_peak": get_attr(policy_summary, "cost_grid_peak", np.nan),
            "sum_cost_grid_ramp": get_attr(policy_summary, "cost_grid_ramp", np.nan),
            "sum_cost_grid_peak_raw": get_attr(policy_summary, "cost_grid_peak_raw", np.nan),
            "sum_cost_grid_ramp_raw": get_attr(policy_summary, "cost_grid_ramp_raw", np.nan),

            # EV deficits
            "sum_ev_departure_deficit_kwh": get_attr(policy_summary, "deficit_kwh_total", np.nan),
            "sum_ev_avoidable_deficit_kwh": get_attr(policy_summary, "deficit_kwh_avoidable", np.nan),
            "sum_ev_unavoidable_deficit_kwh": get_attr(policy_summary, "deficit_kwh_unavoidable", np.nan),
            "sum_ev_departure_departures": get_attr(policy_summary, "ev_departure_departures", np.nan),

            # CityLearn KPIs (episode-end)
            "citylearn_electricity_consumption_total": get_attr(policy_summary, "citylearn_electricity_consumption_total", np.nan),
            "citylearn_carbon_emissions_total": get_attr(policy_summary, "citylearn_carbon_emissions_total", np.nan),
            "citylearn_cost_total": get_attr(policy_summary, "citylearn_cost_total", np.nan),
            "citylearn_zero_net_energy": get_attr(policy_summary, "citylearn_zero_net_energy", np.nan),
            "citylearn_discomfort_proportion": get_attr(policy_summary, "citylearn_discomfort_proportion", np.nan),
            "citylearn_ramping_average": get_attr(policy_summary, "citylearn_ramping_average", np.nan),
            "citylearn_daily_peak_average": get_attr(policy_summary, "citylearn_daily_peak_average", np.nan),
            "citylearn_all_time_peak_average": get_attr(policy_summary, "citylearn_all_time_peak_average", np.nan),

            # Oracle gap
            "oracle_deficit_kwh": to_float(gap.get("oracle_deficit_kwh", np.nan)),
            "policy_deficit_kwh_oracle_eval": to_float(gap.get("policy_deficit_kwh", np.nan)),
            "avoidable_wrt_oracle_kwh": to_float(gap.get("avoidable_wrt_oracle_kwh", np.nan)),
            "avoidable_wrt_oracle_fraction": to_float(gap.get("avoidable_wrt_oracle_fraction", np.nan)),

            "elapsed_sec": elapsed,
        }
        rows.append(r)

        print(f"[{cfg.name}] DONE. sum_cost={r['sum_cost']:.3f}  ev_avoidable={r['sum_ev_avoidable_deficit_kwh']:.6f}  oracle={r['oracle_deficit_kwh']:.3f}")

    df = pd.DataFrame(rows)
    per_seed_csv = out_dir / "per_seed_results.csv"
    df.to_csv(per_seed_csv, index=False)

    # cost "summary" (same as per-seed but kept for your pipeline consistency)
    cost_cols = [
        "sum_cost","sum_cost_ev_departure","sum_cost_grid_peak","sum_cost_grid_ramp",
        "sum_cost_grid_peak_raw","sum_cost_grid_ramp_raw",
        "sum_ev_departure_deficit_kwh","sum_ev_avoidable_deficit_kwh","sum_ev_unavoidable_deficit_kwh",
        "oracle_deficit_kwh","avoidable_wrt_oracle_kwh","avoidable_wrt_oracle_fraction",
    ]
    cost_csv = out_dir / "cost_split_summary.csv"
    df[["model_name"] + cost_cols].to_csv(cost_csv, index=False)

    kpi_cols = [
        "citylearn_electricity_consumption_total","citylearn_carbon_emissions_total","citylearn_cost_total",
        "citylearn_zero_net_energy","citylearn_discomfort_proportion","citylearn_ramping_average",
        "citylearn_daily_peak_average","citylearn_all_time_peak_average",
    ]
    kpi_csv = out_dir / "citylearn_kpis_summary.csv"
    df[["model_name"] + kpi_cols].to_csv(kpi_csv, index=False)

    print("\n" + "="*90)
    print("DONE")
    print(f"Per-seed:  {per_seed_csv}")
    print(f"Cost sum:  {cost_csv}")
    print(f"KPIs:      {kpi_csv}")
    print("="*90)

if __name__ == "__main__":
    main()
