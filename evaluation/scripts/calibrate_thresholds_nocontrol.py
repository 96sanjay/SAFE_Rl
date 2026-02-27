"""
Run no-control baseline (zero actions) to calibrate thresholds
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

from citylearn.citylearn import CityLearnEnv
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.adapters import SingleAgentListAdapter


def make_env():
    schema = os.environ.get("CITYLEARN_SCHEMA", "").strip()
    if not schema:
        raise RuntimeError("CITYLEARN_SCHEMA env var is not set.")
    base = CityLearnEnv(schema=schema, central_agent=True)
    env = SingleAgentListAdapter(base)
    env = CityLearnSafetyEnvV3(env)
    return env


def run_nocontrol(seed: int = 42):
    """Run episode with all-zero actions"""
    os.environ["CITYLEARN_KPI_RUN_NAME"] = f"NOCONTROL_CALIBRATION_seed{seed}"
    
    env = make_env()
    obs, info = env.reset(seed=seed)
    
    action_dim = env.action_space.shape[0]
    zero_action = np.zeros(action_dim, dtype=float)
    
    done = False
    steps = 0
    
    # Collect step data
    building_powers = []
    grid_imports = []
    
    print("Running no-control baseline (zero actions)...")
    
    while not done:
        obs, reward, term, trunc, info = env.step(zero_action)
        done = bool(term or trunc)
        steps += 1
        
        # Extract building power (district total / 17 buildings)
        net_consumption = abs(float(info.get("step_net_consumption_kwh", 0.0)))
        avg_building_power = net_consumption / 17.0
        building_powers.append(avg_building_power)
        
        # Extract grid import
        grid_import = float(info.get("grid_import_kwh", 0.0))
        grid_imports.append(grid_import)
        
        if steps % 1000 == 0:
            print(f"  Step {steps}/8760...")
    
    env.close()
    
    return {
        "building_powers": np.array(building_powers),
        "grid_imports": np.array(grid_imports),
        "steps": steps
    }


def main():
    print("="*70)
    print("THRESHOLD CALIBRATION - No-Control Baseline")
    print("="*70)
    
    # Run no-control
    data = run_nocontrol(seed=42)
    
    building_powers = data["building_powers"]
    grid_imports = data["grid_imports"]
    
    print(f"\nCompleted {data['steps']} steps\n")
    
    # ========================================================================
    # Building Power Statistics
    # ========================================================================
    print("="*70)
    print("[1] BUILDING POWER CAPACITY (per building)")
    print("="*70)
    
    print(f"\nNo-control statistics (district avg ÷ 17):")
    print(f"  Maximum:        {building_powers.max():.2f} kW")
    print(f"  99th percentile: {np.percentile(building_powers, 99):.2f} kW")
    print(f"  97th percentile: {np.percentile(building_powers, 97):.2f} kW")
    print(f"  95th percentile: {np.percentile(building_powers, 95):.2f} kW")
    print(f"  Mean:           {building_powers.mean():.2f} kW")
    print(f"  Median:         {np.median(building_powers):.2f} kW")
    
    p97_building = np.percentile(building_powers, 97)
    print(f"\n✅ RECOMMENDED THRESHOLD: {p97_building:.2f} kW")
    print(f"   (97th percentile → ~3% violations under no-control)")
    
    # ========================================================================
    # Grid Power Statistics
    # ========================================================================
    print("\n" + "="*70)
    print("[2] GRID POWER CAPACITY (district import)")
    print("="*70)
    
    print(f"\nNo-control statistics:")
    print(f"  Maximum:        {grid_imports.max():.2f} kW")
    print(f"  99th percentile: {np.percentile(grid_imports, 99):.2f} kW")
    print(f"  97th percentile: {np.percentile(grid_imports, 97):.2f} kW")
    print(f"  95th percentile: {np.percentile(grid_imports, 95):.2f} kW")
    print(f"  Mean:           {grid_imports.mean():.2f} kW")
    print(f"  Median:         {np.median(grid_imports):.2f} kW")
    
    p97_grid = np.percentile(grid_imports, 97)
    print(f"\n✅ RECOMMENDED THRESHOLD: {p97_grid:.2f} kW")
    print(f"   (97th percentile → ~3% violations under no-control)")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("RECOMMENDED ENV VARS (based on no-control 97th percentile):")
    print("="*70)
    print(f'export CITYLEARN_STEMS_P_BUILDING_MAX="{p97_building:.2f}"')
    print(f'export CITYLEARN_STEMS_P_GRID_MAX="{p97_grid:.2f}"')
    print()
    
    # Save to CSV for reference
    out_dir = os.path.join("runs", "kpi_logs")
    os.makedirs(out_dir, exist_ok=True)
    
    results_df = pd.DataFrame({
        "metric": ["building_power_p97", "grid_power_p97"],
        "value_kw": [p97_building, p97_grid],
        "description": [
            "Per-building power capacity (97th percentile)",
            "District grid import capacity (97th percentile)"
        ]
    })
    
    out_path = os.path.join(out_dir, "nocontrol_threshold_calibration.csv")
    results_df.to_csv(out_path, index=False)
    print(f"Saved calibration results to: {out_path}\n")


if __name__ == "__main__":
    main()
