import os, sys
import numpy as np
import pandas as pd

REPO_ROOT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
sys.path.insert(0, REPO_ROOT)

from scripts.make_env import make_base_env
from citylearn_safe.safety_env import CityLearnSafetyEnv
from scripts.rbc_policy import IntelligentRBC

def main(ev_mode="greedy", seed=42):
    # --- Make env ---
    base_env = make_base_env(central_agent=True)
    env = CityLearnSafetyEnv(base_env)
    agent = IntelligentRBC(env, ev_mode=ev_mode)

    obs, info = env.reset(seed=seed)

    # --- Accumulators ---
    steps = 0
    total_cmdp_cost = 0.0

    # unweighted component sums (as logged)
    sum_ev = sum_soc = sum_bld = sum_grid = 0.0

    # flags (counts)
    cnt_soc = cnt_bld = cnt_grid = 0

    # EV stats
    ev_departures = 0
    ev_def_total = 0.0
    ev_def_avoid = 0.0
    ev_def_unavoid = 0.0

    done = False
    while not done:
        a = agent.predict(obs)
        obs, r, term, trunc, info = env.step(a)
        done = bool(term or trunc)
        steps += 1

        total_cmdp_cost += float(info.get("cost", 0.0))

        sum_ev   += float(info.get("cost_ev_departure", 0.0))
        sum_soc  += float(info.get("cost_stems_battery", 0.0))
        sum_bld  += float(info.get("cost_stems_building_power", 0.0))
        sum_grid += float(info.get("cost_stems_grid_power", 0.0))

        cnt_soc  += int(float(info.get("battery_soc_violation", 0.0)) > 0.0)
        cnt_bld  += int(float(info.get("building_power_violation", 0.0)) > 0.0)
        cnt_grid += int(float(info.get("grid_power_violation", 0.0)) > 0.0)

        ev_departures += int(info.get("ev_departure_departures", 0) or 0)
        ev_def_total  += float(info.get("ev_departure_deficit_kwh", 0.0))
        ev_def_avoid  += float(info.get("ev_avoidable_deficit_kwh", 0.0))
        ev_def_unavoid+= float(info.get("ev_unavoidable_deficit_kwh", 0.0))

    env.close()

    # --- Print report ---
    print("\n=== RBC 1-episode constraint report ===")
    print("ev_mode:", ev_mode, "| seed:", seed)
    print("steps:", steps)
    print("TOTAL CMDP cost:", total_cmdp_cost)
    print("component sums (as logged; NOT reweighted here):")
    print("  EV  :", sum_ev)
    print("  SOC :", sum_soc)
    print("  BLD :", sum_bld)
    print("  GRID:", sum_grid)
    print("violation steps (flags):")
    print("  SOC :", cnt_soc, f"({cnt_soc/steps*100:.2f}%)")
    print("  BLD :", cnt_bld, f"({cnt_bld/steps*100:.2f}%)")
    print("  GRID:", cnt_grid, f"({cnt_grid/steps*100:.2f}%)")
    print("EV departures:", ev_departures)
    print("EV deficits kWh (total / avoidable / unavoidable):",
          ev_def_total, ev_def_avoid, ev_def_unavoid)

    # --- Also parse CSV (sanity) ---
    run_name = os.environ.get("CITYLEARN_KPI_RUN_NAME", "").strip()
    if run_name:
        path = f"runs/kpi_logs/{run_name}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            print("\nCSV sanity:", path, "| rows:", len(df), "| max step:", df["step"].max())
        else:
            print("\nCSV sanity: run_name set but file missing:", path)
    else:
        print("\nCSV sanity: CITYLEARN_KPI_RUN_NAME not set (still logged under default).")

if __name__ == "__main__":
    ev_mode = os.environ.get("RBC_EV_MODE", "greedy")
    seed = int(os.environ.get("RBC_SEED", "42"))
    main(ev_mode=ev_mode, seed=seed)
