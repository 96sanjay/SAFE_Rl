
"""
Default RBC Baseline - FIXED to use CityLearnSafetyEnv backbone

Changes from original:
1. Uses make_base_env() + CityLearnSafetyEnv wrapper (same as OmniSafe)
2. RBC agent now sees the wrapped environment properly
3. KPIs come from info dict (consistent with RL)
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Use the unified backbone
sys.path.insert(0, '/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork')
from scripts.make_env import make_base_env
from citylearn_safe.safety_env import CityLearnSafetyEnv

# RBC import
# try:
#     from citylearn.agents.rbc import BasicRBCAgent
# except ImportError:
#     from citylearn.agents import BasicRBCAgent


from citylearn.agents.rbc import RBC
# OR
from citylearn.agents.rbc import BasicRBC

def main():
    print("=" * 60)
    print(" Running Default-RBC Baseline (UNIFIED BACKBONE)")
    print("=" * 60)

    # Set schema
    schema_path = os.environ.get("CITYLEARN_SCHEMA")
    if not schema_path:
        schema_path = "/home/extra-storage/THESIS/CityLearn/data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"
        os.environ["CITYLEARN_SCHEMA"] = schema_path
    
    os.environ["CITYLEARN_COST_MODE"] = "hinge"
    os.environ["CITYLEARN_INCLUDE_EV_COST"] = "1"

    # Create environment using SAME wrapper as OmniSafe
    print("[INFO] Creating environment with safety wrapper...")
    base_env = make_base_env(central_agent=True)
    env = CityLearnSafetyEnv(
        base_env,
        soc_min=0.0,
        soc_max=0.95,
        cost_mode="hinge"
    )

    # Create RBC agent
    # RBC needs access to unwrapped CityLearn env for building info
    citylearn_env = env._get_citylearn_env()
    agent = BasicRBCAgent(citylearn_env)
    
    print(f"[INFO] Created RBC agent for {len(citylearn_env.buildings)} buildings")

    obs, info = env.reset(seed=42)

    # Output paths
    runs_dir = Path("runs/baselines/default_rbc")
    runs_dir.mkdir(parents=True, exist_ok=True)
    kpi_csv = runs_dir / "kpis.csv"
    summary_file = runs_dir / "summary.json"

    step_rows = []
    done = False
    step = 0

    total_cost = 0.0
    total_reward = 0.0
    total_ev_deficit = 0.0

    print("[INFO] Simulation started...")

    while not done:
        # Get RBC action
        # RBC expects list-of-list for central agent
        obs_for_rbc = [obs] if obs.ndim == 1 else obs
        action_raw = agent.predict(obs_for_rbc)
        
        # Unwrap to single array
        if isinstance(action_raw, list):
            action = action_raw[0] if len(action_raw) == 1 else np.concatenate(action_raw)
        else:
            action = action_raw
        
        action = np.asarray(action, dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Extract KPIs from info dict (same as No-Control)
        row = {
            "step": step + 1,
            "reward": float(reward),
            "cost": info.get("cost", 0.0),
            "cost_building_soc": info.get("cost_building_soc", 0.0),
            "cost_ev_departure": info.get("cost_ev_departure", 0.0),
            
            "soc_mean": info.get("soc_mean", 0.0),
            "soc_min": info.get("soc_min", 0.0),
            "soc_max": info.get("soc_max", 0.0),
            "soc_std": info.get("soc_std", 0.0),
            
            "ev_departure_deficit_kwh": info.get("ev_departure_deficit_kwh", 0.0),
            "ev_avoidable_deficit_kwh": info.get("ev_avoidable_deficit_kwh", 0.0),
            "ev_unavoidable_deficit_kwh": info.get("ev_unavoidable_deficit_kwh", 0.0),
            "ev_departure_departures": info.get("ev_departure_departures", 0),
            
            "battery_abuse_kwh": info.get("battery_abuse_kwh", 0.0),
            "battery_abuse_excess_kwh_equiv": info.get("battery_abuse_excess_kwh_equiv", 0.0),
            "battery_abuse_hours": info.get("battery_abuse_hours", 0.0),
            
            "step_net_consumption_kwh": info.get("step_net_consumption_kwh", 0.0),
            "grid_import_kwh": info.get("grid_import_kwh", 0.0),
            "grid_export_kwh": info.get("grid_export_kwh", 0.0),
            "step_cost": info.get("step_cost", 0.0),
            "step_carbon_kg": info.get("step_carbon_kg", 0.0),
            
            "solar_generation_kwh": info.get("solar_generation_kwh", 0.0),
            "solar_waste_kwh": info.get("solar_waste_kwh", 0.0),
            "thermal_discomfort": info.get("thermal_discomfort", 0.0),
            
            "electricity_price": info.get("electricity_price", 0.0),
            "carbon_intensity": info.get("carbon_intensity", 0.0),
        }

        step_rows.append(row)
        
        total_cost += row["cost"]
        total_reward += row["reward"]
        total_ev_deficit += row["ev_departure_deficit_kwh"]
        
        step += 1
        if step % 2000 == 0:
            print(f"[INFO] Step {step}: cost={total_cost:.2f}, ev_deficit={total_ev_deficit:.1f} kWh")

    # Save KPI timeseries
    print(f"[INFO] Writing KPIs to {kpi_csv}")
    df = pd.DataFrame(step_rows)
    df.to_csv(kpi_csv, index=False)

    # Summary statistics
    summary = {
        "agent": "Default-RBC",
        "total_steps": step,
        "total_cost": float(total_cost),
        "total_reward": float(total_reward),
        "total_ev_deficit_kwh": float(total_ev_deficit),
        
        "soc_violation_count": int((df["soc_max"] > 0.95).sum()),
        "ev_failure_count": int((df["ev_departure_deficit_kwh"] > 0.001).sum()),
        "battery_abuse_total_kwh": float(df["battery_abuse_kwh"].sum()),
        
        "avg_soc_mean": float(df["soc_mean"].mean()),
        "avg_cost_per_step": float(total_cost / step) if step > 0 else 0.0,
    }

    if "citylearn_cost_total" in info:
        summary["citylearn_cost_total"] = float(info["citylearn_cost_total"])
        summary["citylearn_carbon_emissions_total"] = float(info.get("citylearn_carbon_emissions_total", 0.0))
        summary["citylearn_ramping_average"] = float(info.get("citylearn_ramping_average", 0.0))
        summary["citylearn_daily_peak_average"] = float(info.get("citylearn_daily_peak_average", 0.0))

    import json
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total steps: {step}")
    print(f"Total cost: {total_cost:.2f}")
    print(f"Total EV deficit: {total_ev_deficit:.1f} kWh")
    print(f"SoC>0.95 violations: {summary['soc_violation_count']}")
    print(f"Battery abuse total: {summary['battery_abuse_total_kwh']:.1f} kWh")
    print("=" * 60)
    print(f"\n✅ Results saved to {runs_dir}")

    env.close()


if __name__ == "__main__":
    main()