
"""
Intelligent Custom RBC - FIXED action name unwrapping
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork')
from scripts.make_env import make_base_env
from citylearn_safe.safety_env import CityLearnSafetyEnv


class IntelligentRBC:
    """Intelligent Rule-Based Controller with EV priority."""
    
    def __init__(self, env):
        self.env = env
        self.action_dim = env.action_space.shape[0]
        
        # Get action names - UNWRAP list-of-list!
        if hasattr(env, 'action_names'):
            names = env.action_names
            if isinstance(names, list) and len(names) == 1 and isinstance(names[0], list):
                self.action_names = names[0]  # Unwrap central agent list
            else:
                self.action_names = names
        else:
            self.action_names = []
        
        print(f"[DEBUG] Action names (first 5): {self.action_names[:5]}")
        print(f"[DEBUG] Action names (last 5): {self.action_names[-5:]}")
        
        # Identify action indices
        self.battery_indices = self._find_battery_indices()
        self.ev_indices = self._find_ev_indices()
        self.hvac_indices = self._find_hvac_indices()
        
        print(f"[IntelligentRBC] Initialized:")
        print(f"  Total action dim: {self.action_dim}")
        print(f"  Battery indices: {self.battery_indices}")
        print(f"  EV indices: {self.ev_indices}")
        print(f"  HVAC indices: {self.hvac_indices}")
        
        self._step_count = 0
    
    def _find_battery_indices(self):
        """Find action indices for building batteries."""
        indices = []
        for i, name in enumerate(self.action_names):
            name_lower = str(name).lower()
            # Batteries are 'electrical_storage' but NOT 'electric_vehicle_storage'
            if 'electrical_storage' in name_lower and 'vehicle' not in name_lower:
                indices.append(i)
        return indices
    
    def _find_ev_indices(self):
        """Find action indices for EV chargers."""
        indices = []
        for i, name in enumerate(self.action_names):
            name_lower = str(name).lower()
            if 'electric_vehicle' in name_lower or 'charger' in name_lower:
                indices.append(i)
        return indices
    
    def _find_hvac_indices(self):
        """Find action indices for HVAC."""
        indices = []
        for i, name in enumerate(self.action_names):
            name_lower = str(name).lower()
            if any(x in name_lower for x in ['heating', 'cooling', 'dhw', 'hot_water']):
                indices.append(i)
        return indices
    
    def _battery_action(self, hour):
        """
        Simple time-based battery strategy.
        - Charge during solar hours (10-16)
        - Discharge during peak hours (17-21)
        - Neutral otherwise
        """
        if 10 <= hour <= 16:
            return 0.8  # Charge (solar hours)
        elif 17 <= hour <= 21:
            return -0.6  # Discharge (peak hours)
        else:
            return 0.0  # Neutral
    
    def predict(self, obs):
        """Generate action for current observation."""
        action = np.zeros(self.action_dim, dtype=np.float32)
        
        hour = int(self._step_count % 24)
        
        # 1. EV CHARGING: Greedy (always charge at max)
        for ev_idx in self.ev_indices:
            if ev_idx < self.action_dim:
                action[ev_idx] = 1.0  # Maximum charge
        
        # 2. BATTERY: Time-based strategy
        battery_val = self._battery_action(hour)
        for batt_idx in self.battery_indices:
            if batt_idx < self.action_dim:
                action[batt_idx] = battery_val
        
        # 3. HVAC: Neutral (let building handle)
        for hvac_idx in self.hvac_indices:
            if hvac_idx < self.action_dim:
                action[hvac_idx] = 0.0
        
        self._step_count += 1
        return action


def main():
    print("=" * 60)
    print(" Intelligent RBC Baseline - FIXED")
    print("=" * 60)

    schema_path = os.environ.get("CITYLEARN_SCHEMA")
    if not schema_path:
        schema_path = "/home/extra-storage/THESIS/CityLearn/data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"
        os.environ["CITYLEARN_SCHEMA"] = schema_path
    
    os.environ["CITYLEARN_COST_MODE"] = "hinge"
    os.environ["CITYLEARN_INCLUDE_EV_COST"] = "1"

    print("[INFO] Creating environment...")
    base_env = make_base_env(central_agent=True)
    env = CityLearnSafetyEnv(base_env, soc_min=0.0, soc_max=0.95, cost_mode="hinge")

    agent = IntelligentRBC(env)
    
    obs, info = env.reset(seed=42)

    runs_dir = Path("runs/baselines/intelligent_rbc")
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
        action = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        row = {
            "step": step + 1,
            "reward": float(reward),
            "cost": info.get("cost", 0.0),
            "cost_building_soc": info.get("cost_building_soc", 0.0),
            "cost_ev_departure": info.get("cost_ev_departure", 0.0),
            
            "soc_mean": info.get("soc_mean", 0.0),
            "soc_max": info.get("soc_max", 0.0),
            "soc_min": info.get("soc_min", 0.0),
            
            "ev_departure_deficit_kwh": info.get("ev_departure_deficit_kwh", 0.0),
            "ev_avoidable_deficit_kwh": info.get("ev_avoidable_deficit_kwh", 0.0),
            "ev_unavoidable_deficit_kwh": info.get("ev_unavoidable_deficit_kwh", 0.0),
            
            "battery_abuse_kwh": info.get("battery_abuse_kwh", 0.0),
            "grid_import_kwh": info.get("grid_import_kwh", 0.0),
            "grid_export_kwh": info.get("grid_export_kwh", 0.0),
            "step_cost": info.get("step_cost", 0.0),
        }

        step_rows.append(row)
        total_cost += row["cost"]
        total_reward += row["reward"]
        total_ev_deficit += row["ev_departure_deficit_kwh"]
        
        step += 1
        if step % 2000 == 0:
            print(f"[INFO] Step {step}: cost={total_cost:.2f}, soc_max={info.get('soc_max', 0):.3f}, ev_deficit={total_ev_deficit:.1f}")

    print(f"[INFO] Writing KPIs to {kpi_csv}")
    df = pd.DataFrame(step_rows)
    df.to_csv(kpi_csv, index=False)

    summary = {
        "agent": "Intelligent-RBC",
        "total_steps": step,
        "total_cost": float(total_cost),
        "total_ev_deficit_kwh": float(total_ev_deficit),
        "soc_violation_count": int((df["soc_max"] > 0.95).sum()),
        "battery_abuse_total": float(df["battery_abuse_kwh"].sum()),
    }

    import json
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Total cost: {total_cost:.2f}")
    print(f"EV deficit: {total_ev_deficit:.1f} kWh")
    print(f"SoC violations: {summary['soc_violation_count']}")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()