#!/usr/bin/env python3
"""Quick eval of the dumb greedy baseline (charge everything always)."""
import os, sys, numpy as np

PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

from eval_all_proper import (
    parse_exports, build_env, evaluate_run, suppress_stdout, log
)

SEED = 42
np.random.seed(SEED)

# Verified action mapping from schema_5buildings.json:
# [0] electrical_storage           (Bldg 1 battery)
# [1] electric_vehicle_charger_1_1 (Bldg 1 EV, 11 kW)
# [2] washing_machine_1            (Bldg 1 washer)
# [3] electrical_storage           (Bldg 2 battery)
# [4] electrical_storage           (Bldg 3 battery)
# [5] electrical_storage           (Bldg 4 battery)
# [6] electric_vehicle_charger_4_1 (Bldg 4 EV, 22 kW)
# [7] electrical_storage           (Bldg 5 battery)
# [8] electric_vehicle_charger_5_1 (Bldg 5 EV, 7.4 kW)

BATT_INDICES = [0, 3, 4, 5, 7]
EV_INDICES = [1, 6, 8]
WASHER_INDEX = 2

def main():
    script_path = os.path.join(PROJECT, "run_r19_ablation.sh")
    env_vars = parse_exports(script_path)
    meta = {
        "saute": True, "action_mask": False, "batt_clamp": False,
        "bc_warmstart": False, "pid_lagrange": True, "curriculum": True,
        "lambda_ev": 0, "ev_guard": 0, "v2g_context": 0, "load_shift": 0,
        "price_arb": 0, "solar_store": 0, "headroom": 0, "ev_solar": 0,
        "ev_slack_arb": 0, "grid_penalty": 0,
    }

    log("--- greedy_ev_dumb (charge EVs max + batteries 0.5 always) ---")
    with suppress_stdout():
        env = build_env(env_vars, use_action_mask=False)
    act_dim = env.action_space.shape[0]
    log(f"  act_dim={act_dim}")
    log(f"  BATT indices: {BATT_INDICES}")
    log(f"  EV indices: {EV_INDICES}")

    def dumb_greedy_policy(obs):
        action = np.zeros(act_dim)
        for i in EV_INDICES:
            action[i] = 1.0   # full EV charge always
        for i in BATT_INDICES:
            action[i] = 0.5   # constant battery charge from grid
        # washer and other indices stay 0
        return action

    test = dumb_greedy_policy(np.zeros(10))
    log(f"  Example action: {test}")

    result = evaluate_run("greedy_ev_dumb", env, dumb_greedy_policy, env_vars, meta)

    log(f"\n{'='*60}")
    log(f"RESULTS: Dumb Greedy (charge everything always)")
    log(f"{'='*60}")
    log(f"  C0: {result.c0_violated}/{result.c0_total_departures} ({result.c0_violation_pct:.1f}%)")
    log(f"  C2: {result.c2_violation_pct:.1f}%")
    log(f"  C3: {result.c3_violations}/{result.c3_total} ({result.c3_violation_pct:.1f}%)")
    log(f"  C4: {result.c4_violations}/{result.c4_total} ({result.c4_violation_pct:.1f}%)")
    log(f"  Import: {result.total_import_kwh:.0f} kWh")
    log(f"  Export: {result.total_export_kwh:.0f} kWh")
    log(f"  Peak NEC: {result.peak_nec_kw:.1f} kW")
    log(f"  Ramping: {result.ramping_kwh:.0f} kWh")
    log(f"  Elec Cost: {result.electricity_cost:.0f}")
    log(f"  Batt: charge={result.batt_charge_pct:.1f}% discharge={result.batt_discharge_pct:.1f}% idle={result.batt_idle_pct:.1f}%")
    log(f"  EV: charge={result.ev_charge_pct:.1f}% v2g={result.ev_v2g_pct:.1f}% v2g_kwh={result.ev_total_v2g_kwh:.0f}")
    log(f"  Time: {result.eval_seconds:.1f}s")

    log(f"\n{'='*60}")
    log(f"COMPARISON (from master_eval_table.md):")
    log(f"{'='*60}")
    log(f"{'Controller':<22} {'C0%':>6} {'C3%':>6} {'C4%':>6} {'Import':>8} {'Cost':>8}")
    log(f"{'-'*58}")
    log(f"{'Dumb Greedy (new)':<22} {result.c0_violation_pct:>5.1f}% {result.c3_violation_pct:>5.1f}% {result.c4_violation_pct:>5.1f}% {result.total_import_kwh:>7.0f} {result.electricity_cost:>7.0f}")
    log(f"{'Greedy EV (old)':<22} {'0.7%':>6} {'9.4%':>6} {'20.2%':>6} {'57907':>8} {'9877':>8}")
    log(f"{'SmartV2GRBC':<22} {'0.7%':>6} {'13.2%':>6} {'31.0%':>6} {'66470':>8} {'11081':>8}")
    log(f"{'PPO r25b':<22} {'0.7%':>6} {'8.5%':>6} {'20.0%':>6} {'51758':>8} {'8893':>8}")
    log(f"{'Zero Action':<22} {'100%':>6} {'3.1%':>6} {'1.8%':>6} {'25192':>8} {'4313':>8}")

if __name__ == "__main__":
    main()
