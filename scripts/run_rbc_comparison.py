"""
RBC Baseline Comparison: Greedy vs Time-Based EV Charging

Runs BOTH modes and generates comparison report for professor
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

REPO_ROOT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
sys.path.insert(0, REPO_ROOT)

from scripts.make_env import make_base_env
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.extractors_v3 import unwrap_to_raw_citylearn_env, current_time_index


def _unwrap_action_names(names):
    if isinstance(names, list) and len(names) == 1 and isinstance(names[0], list):
        return names[0]
    return names


def _norm_id(x) -> str:
    if isinstance(x, (bytes, np.bytes_)):
        x = x.decode("utf-8", errors="ignore")
    s = str(x).strip()
    if s.startswith("b'") and s.endswith("'"):
        s = s[2:-1]
    return s.strip()


def _is_valid_ev_id(x) -> bool:
    s = _norm_id(x)
    if s == "" or s.lower() in ("nan", "none"):
        return False
    return True


class IntelligentRBC:
    """RBC with configurable EV charging mode"""

    def __init__(self, env: CityLearnSafetyEnvV3, ev_mode: str = "plugged_only"):
        self.env = env
        self.ev_mode = ev_mode

        names = _unwrap_action_names(env.action_names)
        self.action_names = names

        if hasattr(env.action_space, "shape"):
            self.action_dim = int(env.action_space.shape[0])
        else:
            self.action_dim = len(names)

        self.battery_indices = [i for i, n in enumerate(names) if str(n).strip().lower() == "electrical_storage"]
        self.ev_indices = [i for i, n in enumerate(names) if "electric_vehicle_storage_charger" in str(n).lower()]
        self.cooling_indices = [i for i, n in enumerate(names) if str(n).strip().lower() == "cooling_storage"]
        self.heating_indices = [i for i, n in enumerate(names) if str(n).strip().lower() == "heating_storage"]

    def _battery_action(self, hour: int) -> float:
        return 0.8 if 10 <= hour <= 16 else (-0.6 if 17 <= hour <= 21 else 0.0)

    def _get_indoor_temps(self, raw_env: Any) -> np.ndarray:
        temps = []
        for b in getattr(raw_env, "buildings", []) or []:
            t_idx = int(getattr(raw_env, "time_step", 0))
            if t_idx < 0:
                t_idx = 0
            try:
                temp_arr = np.asarray(b.indoor_dry_bulb_temperature, dtype=float)
                if temp_arr.ndim == 1 and 0 <= t_idx < len(temp_arr):
                    temps.append(float(temp_arr[t_idx]))
                else:
                    temps.append(22.0)
            except:
                temps.append(22.0)
        return np.array(temps)

    def _charger_connected_now(self, raw_env: Any, action_name: str) -> bool:
        s = str(action_name).strip().lower()
        if "electric_vehicle_storage_charger_" not in s:
            return False

        suffix = s.split("electric_vehicle_storage_charger_", 1)[1]
        charger_id = f"charger_{suffix}"

        t_state = int(getattr(raw_env, "time_step", 0))
        if t_state < 0:
            t_state = 0

        for b in getattr(raw_env, "buildings", []) or []:
            for ch in getattr(b, "electric_vehicle_chargers", []) or []:
                cid = getattr(ch, "charger_id", getattr(ch, "name", None))
                if _norm_id(cid) != charger_id:
                    continue

                sim = getattr(ch, "charger_simulation", getattr(ch, "_Charger__charger_simulation", None))
                if sim is None:
                    return False

                try:
                    state = np.asarray(getattr(sim, "_electric_vehicle_charger_state"), dtype=float)
                    ev_id = np.asarray(getattr(sim, "_electric_vehicle_id"))
                except Exception:
                    return False

                if state.ndim != 1 or t_state >= len(state):
                    return False

                st = float(state[t_state])
                eid = ev_id[t_state]
                return (st == 1.0) and _is_valid_ev_id(eid)

        return False

    def predict(self, obs) -> np.ndarray:
        a = np.zeros(self.action_dim, dtype=np.float32)

        raw = unwrap_to_raw_citylearn_env(self.env)
        hour = int(current_time_index(raw) % 24)

        # EV charging strategy
        if self.ev_mode == "always":
            for i in self.ev_indices:
                a[i] = 1.0  # Always try to charge
        elif self.ev_mode == "plugged_only":
            for i in self.ev_indices:
                a[i] = 1.0 if self._charger_connected_now(raw, self.action_names[i]) else 0.0

        # Battery (same for both)
        batt = float(self._battery_action(hour))
        for i in self.battery_indices:
            a[i] = batt

        # Temperature (same for both)
        temps = self._get_indoor_temps(raw)
        for i in self.cooling_indices:
            a[i] = 0.5 if temps[i % len(temps)] > 24.0 else 0.0
        for i in self.heating_indices:
            a[i] = 0.5 if temps[i % len(temps)] < 20.0 else 0.0

        return a


def run_baseline(ev_mode: str, output_dir: Path):
    """Run one baseline configuration"""
    
    print("\n" + "=" * 90)
    print(f"RUNNING: EV Mode = {ev_mode}")
    print("=" * 90)
    
    base_env = make_base_env(central_agent=True)
    env = CityLearnSafetyEnvV3(base_env, soc_min=0.0, soc_max=0.95, cost_mode="hinge")
    agent = IntelligentRBC(env, ev_mode=ev_mode)

    obs, info = env.reset(seed=42)

    rows = []
    step = 0
    done = False

    while not done:
        action = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        row = {"step": step + 1}
        row.update(info)
        rows.append(row)

        if (step + 1) % 1000 == 0:
            print(f"  Step {step + 1:5d}")

        step += 1

    # Save full CSV
    df = pd.DataFrame(rows)
    out_csv = output_dir / f"kpis_{ev_mode}.csv"
    df.to_csv(out_csv, index=False)
    
    print(f"\n✅ Saved: {out_csv}")
    print(f"   {df.shape[0]} rows × {df.shape[1]} columns")
    
    return df


def generate_comparison_report(df_greedy: pd.DataFrame, df_always: pd.DataFrame, output_dir: Path):
    """Generate side-by-side comparison for professor"""
    
    print("\n" + "=" * 90)
    print("GENERATING COMPARISON REPORT")
    print("=" * 90)
    
    # Extract final episode metrics
    metrics_to_compare = [
        # CityLearn KPIs
        "citylearn_electricity_consumption_total",
        "citylearn_carbon_emissions_total",
        "citylearn_cost_total",
        "citylearn_zero_net_energy",
        "citylearn_discomfort_proportion",
        "citylearn_ramping_average",
        "citylearn_daily_peak_average",
        "citylearn_all_time_peak_average",
        
        # Constraint violations
        "cost",
        "cost_building_soc",
        "cost_ev_departure",
        "cost_ev_departure_agent_controllable_v3",
        "cost_ev_departure_uncontrollable_v3",
        
        # Energy metrics
        "ev_departure_deficit_kwh",
        "ev_avoidable_deficit_kwh",
        "ev_unavoidable_deficit_kwh",
        "ev_departure_departures",
        
        # Comfort & abuse
        "thermal_discomfort",
        "battery_abuse_kwh",
        "solar_waste_kwh",
    ]
    
    comparison = []
    
    for metric in metrics_to_compare:
        if metric in df_greedy.columns and metric in df_always.columns:
            # Get final value (or sum for cumulative metrics)
            if 'citylearn' in metric:
                val_greedy = df_greedy[metric].iloc[-1]
                val_always = df_always[metric].iloc[-1]
            else:
                val_greedy = df_greedy[metric].sum()
                val_always = df_always[metric].sum()
            
            diff = val_always - val_greedy
            pct_change = (diff / val_greedy * 100) if val_greedy != 0 else 0.0
            
            comparison.append({
                "Metric": metric,
                "Greedy (plugged_only)": val_greedy,
                "Always (time-based)": val_always,
                "Difference": diff,
                "% Change": pct_change,
            })
    
    df_comp = pd.DataFrame(comparison)
    
    # Save comparison
    comp_csv = output_dir / "comparison_greedy_vs_always.csv"
    df_comp.to_csv(comp_csv, index=False)
    
    print(f"\n✅ Comparison saved: {comp_csv}")
    
    # Print summary
    print("\n" + "=" * 90)
    print("SUMMARY COMPARISON")
    print("=" * 90)
    
    print(f"\n{'Metric':<50} {'Greedy':>15} {'Always':>15} {'Diff':>15}")
    print("-" * 95)
    
    for _, row in df_comp.iterrows():
        metric = row['Metric']
        greedy = row['Greedy (plugged_only)']
        always = row['Always (time-based)']
        diff = row['Difference']
        
        if 'citylearn' in metric or 'cost' in metric:
            print(f"{metric:<50} {greedy:>15.4f} {always:>15.4f} {diff:>15.4f}")
    
    print("=" * 90 + "\n")
    
    return df_comp


def main():
    schema_path = os.environ.get("CITYLEARN_SCHEMA") or \
        "/home/extra-storage/THESIS/CityLearn/data/datasets/citylearn_challenge_2022_phase_all_plus_evs_WITH_TEMP_CONTROL/schema.json"
    os.environ["CITYLEARN_SCHEMA"] = schema_path
    
    # Output directory
    runs_dir = Path("runs/baselines/rbc_comparison")
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 90)
    print("RBC BASELINE COMPARISON: GREEDY vs TIME-BASED EV CHARGING")
    print("=" * 90)
    print(f"Output directory: {runs_dir}")
    print("=" * 90)
    
    # Run both configurations
    df_greedy = run_baseline("plugged_only", runs_dir)
    df_always = run_baseline("always", runs_dir)
    
    # Generate comparison
    df_comp = generate_comparison_report(df_greedy, df_always, runs_dir)
    
    print("\n" + "=" * 90)
    print("✅ COMPLETE! Files generated:")
    print("=" * 90)
    print(f"  1. {runs_dir}/kpis_plugged_only.csv")
    print(f"  2. {runs_dir}/kpis_always.csv")
    print(f"  3. {runs_dir}/comparison_greedy_vs_always.csv")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
