#!/usr/bin/env python3
"""
Batch evaluation of ALL 5-building runs using eval_verified.py.
Produces a master comparison table sorted by C0 violation rate.

Usage:
    python eval_all_5bld.py [--dry-run] [--only NAME] [--skip-existing]
"""
import subprocess
import json
import os
import sys
import time
import argparse
from collections import OrderedDict

PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
PYTHON = "/home/christmas/miniconda3/envs/citylearn/bin/python"
EVAL_SCRIPT = os.path.join(PROJECT, "eval_verified.py")
RESULTS_DIR = os.path.join(PROJECT, "eval_results", "batch_5bld")

# =====================================================================
# All 5-building runs with act_dim=9, 40+ epochs
# Format: (short_name, epochs, checkpoint_path, algo)
# Verified: all have act_dim=9 (5-building compatible)
# =====================================================================
RUNS = [
    # --- Recent PPOLagMulti runs (obs=199, act=9) ---
    ("r22_ppo_ep120", 120,
     "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/runs/r22_ppo/5bld/PPOLagMulti-{CityLearnSafety-V2G-v2}/seed-042-2026-03-15-03-55-41/torch_save/epoch-120.pt",
     "ppo"),
    ("r21_ppo_ep89", 89,
     "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/runs/r21_ppo/5bld/PPOLagMulti-{CityLearnSafety-V2G-v2}/seed-042-2026-03-13-04-07-53/torch_save/epoch-89.pt",
     "ppo"),
    ("r23_ppo_ep89", 89,
     "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/runs/r23_ppo/5bld/PPOLagMulti-{CityLearnSafety-V2G-v2}/seed-042-2026-03-15-04-32-16/torch_save/epoch-89.pt",
     "ppo"),
    ("r29_mask_simple_ep100", 100,
     "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/runs/r29_mask_simple/5bld/PPOLagMulti-{CityLearnSafety-V2G-v2}/seed-042-2026-03-19-06-31-04/torch_save/epoch-100.pt",
     "ppo"),
    ("ablation_mask_ep89", 89,
     "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/runs/ablation_mask_treatment_s42/5bld/PPOLagMulti-{CityLearnSafety-V2G-v2}/seed-042-2026-03-21-07-44-08/torch_save/epoch-89.pt",
     "ppo"),
    ("ablation_single_lag_ep89", 89,
     "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/runs/ablation_single_lag/5bld/PPOLag-{CityLearnSafety-V2G-v2}/seed-000-2026-03-20-06-48-52/torch_save/epoch-89.pt",
     "ppo"),

    # --- Multi-lag v3 (obs=218, act=9) ---
    ("multi_lag_v3a_ep100", 100,
     "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/runs/multi_lag_v3a/5bld/PPOLagMulti-{CityLearnSafety-V2G-v2}/seed-042-2026-03-09-03-57-52/torch_save/epoch-100.pt",
     "ppo"),
    ("multi_lag_v3b_ep100", 100,
     "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/runs/multi_lag_v3b/5bld/PPOLagMulti-{CityLearnSafety-V2G-v2}/seed-042-2026-03-09-07-35-11/torch_save/epoch-100.pt",
     "ppo"),

    # --- Reward tuning runs (obs=199, act=9) ---
    ("r19_ablation_ep80", 80,
     "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/runs/r19_ablation/5bld/PPOLagMulti-{CityLearnSafety-V2G-v2}/seed-042-2026-03-12-20-01-15/torch_save/epoch-80.pt",
     "ppo"),
    ("r24_solar_store_ep80", 80,
     "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/runs/r24_solar_store/5bld/PPOLagMulti-{CityLearnSafety-V2G-v2}/seed-042-2026-03-17-04-31-09/torch_save/epoch-80.pt",
     "ppo"),
    ("r25a_batt_solar_ep80", 80,
     "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/runs/r25a_batt_solar/5bld/PPOLagMulti-{CityLearnSafety-V2G-v2}/seed-042-2026-03-17-17-01-55/torch_save/epoch-80.pt",
     "ppo"),
    ("r25b_ev_slack_ep80", 80,
     "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/runs/r25b_ev_slack_arb/5bld/PPOLagMulti-{CityLearnSafety-V2G-v2}/seed-042-2026-03-17-17-01-55/torch_save/epoch-80.pt",
     "ppo"),
    ("r25c_v2g_reduce_ep80", 80,
     "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/runs/r25c_v2g_reduce/5bld/PPOLagMulti-{CityLearnSafety-V2G-v2}/seed-042-2026-03-17-17-01-55/torch_save/epoch-80.pt",
     "ppo"),
    ("r26_headroom_ep80", 80,
     "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/runs/r26_headroom/5bld/PPOLagMulti-{CityLearnSafety-V2G-v2}/seed-042-2026-03-17-19-56-11/torch_save/epoch-80.pt",
     "ppo"),
    ("r27_price_arb_ep80", 80,
     "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/runs/r27_price_arb/5bld/PPOLagMulti-{CityLearnSafety-V2G-v2}/seed-042-2026-03-18-18-54-27/torch_save/epoch-80.pt",
     "ppo"),

    # --- Older multi-lag runs (obs=198-199, act=9) ---
    ("multi_lag_r4_ep100", 100,
     "/media/christmas/KODAK/Safe-CityLearn-Fork/runs/multi_lag_5b_r4/PPOLag-{CityLearnSafety-V2G-v2-multilag}/seed-000-2026-02-27-20-47-22/torch_save/epoch-100.pt",
     "ppo"),
    ("multi_lag_r5_ep100", 100,
     "/media/christmas/KODAK/Safe-CityLearn-Fork/runs/multi_lag_5b_r5/PPOLag-{CityLearnSafety-V2G-v2-multilag}/seed-000-2026-02-28-08-05-39/torch_save/epoch-100.pt",
     "ppo"),
    ("multi_lag_r8_ep100", 100,
     "/media/christmas/KODAK/Safe-CityLearn-Fork/runs/multi_lag_5b_r8/PPOLag-{CityLearnSafety-V2G-v2-multilag}/seed-000-2026-02-28-20-14-53/torch_save/epoch-100.pt",
     "ppo"),
    ("multi_lag_r9_ep100", 100,
     "/media/christmas/KODAK/Safe-CityLearn-Fork/runs/multi_lag_5b_r9/PPOLag-{CityLearnSafety-V2G-v2-multilag}/seed-000-2026-03-01-08-55-31/torch_save/epoch-100.pt",
     "ppo"),
    ("multi_lag_r10_ep100", 100,
     "/media/christmas/KODAK/Safe-CityLearn-Fork/runs/multi_lag_5b_r10/PPOLag-{CityLearnSafety-V2G-v2-multilag}/seed-000-2026-03-01-21-53-24/torch_save/epoch-100.pt",
     "ppo"),
    ("multi_lag_r11_ep100", 100,
     "/media/christmas/KODAK/Safe-CityLearn-Fork/runs/multi_lag_5b_r11/PPOLag-{CityLearnSafety-V2G-v2-multilag}/seed-000-2026-03-02-02-55-19/torch_save/epoch-100.pt",
     "ppo"),

    # --- C1+C2 only (obs=207, act=9) ---
    ("c1c2_only_r1_ep100", 100,
     "/media/christmas/KODAK/Safe-CityLearn-Fork/runs/c1c2_only_5b_r1/PPOLag-{CityLearnSafety-V2G-v2-multilag}/seed-000-2026-03-03-21-14-38/torch_save/epoch-100.pt",
     "ppo"),

    # --- CR-MOPO (obs=199-207, act=9) ---
    ("cr_mopo_r1_ep99", 99,
     "/media/christmas/KODAK/Safe-CityLearn-Fork/runs/cr_mopo_5b_r1/epoch-99.pt",
     "ppo"),
    ("cr_mopo_r6_ep99", 99,
     "/media/christmas/KODAK/Safe-CityLearn-Fork/runs/cr_mopo_5b_r6/epoch-99.pt",
     "ppo"),
]


def run_eval(name, ckpt_path, algo, results_dir, dry_run=False):
    """Run eval_verified.py for one checkpoint. Returns (success, json_path, elapsed)."""
    out_path = os.path.join(results_dir, f"{name}.json")

    cmd = [
        PYTHON, EVAL_SCRIPT,
        "--checkpoint", ckpt_path,
        "--schema", "5bld",
        "--months", "12",
        "--algo", algo,
        "--no-determinism-check",
        "--output", out_path,
    ]

    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return True, out_path, 0.0

    print(f"  CMD: {PYTHON} eval_verified.py --checkpoint .../{os.path.basename(ckpt_path)} "
          f"--schema 5bld --months 12 --algo {algo} --output {name}.json")

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,  # 15 min max per run
            cwd=PROJECT,
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"  FAILED (return code {result.returncode}, {elapsed:.0f}s)")
            # Save stderr for debugging
            err_path = os.path.join(results_dir, f"{name}_ERROR.txt")
            with open(err_path, "w") as f:
                f.write(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n")
            print(f"  Error log: {err_path}")
            # Print last 5 lines of stderr
            err_lines = result.stderr.strip().split("\n")
            for line in err_lines[-5:]:
                print(f"    {line}")
            return False, out_path, elapsed

        print(f"  OK ({elapsed:.0f}s)")
        return True, out_path, elapsed

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"  TIMEOUT after {elapsed:.0f}s")
        return False, out_path, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  EXCEPTION: {e}")
        return False, out_path, elapsed


def load_results(json_path):
    """Load eval results from JSON file."""
    if not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        return json.load(f)


def print_master_table(all_results):
    """Print a formatted comparison table sorted by C0 violation rate."""
    if not all_results:
        print("\nNo results to display.")
        return

    # Sort by C0 violation %, then by total_reward descending
    sorted_results = sorted(
        all_results,
        key=lambda x: (
            x["metrics"].get("c0_violation_pct", 999),
            -x["metrics"].get("total_reward", 0),
        ),
    )

    # Header
    print("\n" + "=" * 180)
    print("MASTER COMPARISON TABLE -- 5-Building Runs (12-month eval, seed=42)")
    print("=" * 180)

    header = (
        f"{'Run':<30s} {'Ep':>4s} "
        f"{'TotRew':>8s} {'MeanRew':>8s} "
        f"{'C0 Viol%':>8s} {'C0 Deps':>7s} "
        f"{'C3 Viol%':>8s} {'C4 Viol%':>8s} "
        f"{'EV Chg%':>7s} {'V2G%':>5s} {'V2GPk%':>6s} "
        f"{'BtCyc%':>6s} {'PrCorr':>7s} "
        f"{'PkNEC':>7s} {'C0Cost':>7s} {'TotCost':>8s}"
    )
    print(header)
    print("-" * 180)

    for r in sorted_results:
        m = r["metrics"]
        name = r["name"]
        epochs = r["epochs"]

        # Battery cycling percentage
        cyc_days = m.get("batt_cycling_days", 0)
        cyc_total = m.get("batt_cycling_days_total", 1)
        cyc_pct = 100.0 * cyc_days / max(cyc_total, 1)

        # Peak NEC (C4 p95)
        peak_nec = m.get("c4_p95_abs_nec", 0.0)

        row = (
            f"{name:<30s} {epochs:>4d} "
            f"{m.get('total_reward', 0):>8.1f} {m.get('mean_reward', 0):>8.4f} "
            f"{m.get('c0_violation_pct', 0):>8.1f} {m.get('c0_total_departures', 0):>7d} "
            f"{m.get('c3_violation_pct', 0):>8.2f} {m.get('c4_violation_pct', 0):>8.2f} "
            f"{m.get('ev_charge_pct', 0):>7.1f} {m.get('ev_discharge_pct', 0):>5.1f} "
            f"{m.get('ev_v2g_peak_pct', 0):>6.1f} "
            f"{cyc_pct:>6.1f} {m.get('batt_price_corr', 0):>+7.3f} "
            f"{peak_nec:>7.2f} {m.get('c0_cost', 0):>7.1f} {m.get('total_cost', 0):>8.1f}"
        )
        print(row)

    print("-" * 180)
    print(f"Total runs evaluated: {len(sorted_results)}")

    # Also print a condensed version focusing on the key tradeoff
    print("\n" + "=" * 120)
    print("KEY TRADEOFFS (sorted by C0 violation %)")
    print("=" * 120)
    header2 = (
        f"{'Run':<30s} "
        f"{'C0 Viol%':>8s} {'C0 MeanDef':>10s} "
        f"{'C3 Viol%':>8s} {'C4 Viol%':>8s} "
        f"{'TotRew':>8s} {'EV MeanAct':>10s} "
        f"{'BattSolAvg':>10s} {'BattPkAvg':>9s}"
    )
    print(header2)
    print("-" * 120)
    for r in sorted_results:
        m = r["metrics"]
        row2 = (
            f"{r['name']:<30s} "
            f"{m.get('c0_violation_pct', 0):>8.1f} "
            f"{m.get('c0_mean_deficit_violated', 0):>10.4f} "
            f"{m.get('c3_violation_pct', 0):>8.2f} "
            f"{m.get('c4_violation_pct', 0):>8.2f} "
            f"{m.get('total_reward', 0):>8.1f} "
            f"{m.get('ev_mean_action_connected', 0):>+10.4f} "
            f"{m.get('batt_solar_avg', 0):>+10.4f} "
            f"{m.get('batt_peak_avg', 0):>+9.4f}"
        )
        print(row2)
    print("-" * 120)


def main():
    parser = argparse.ArgumentParser(description="Batch eval all 5-building runs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running them")
    parser.add_argument("--only", type=str, default=None,
                        help="Only run eval for this specific run name (substring match)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip runs that already have result JSON files")
    parser.add_argument("--table-only", action="store_true",
                        help="Only print the table from existing results (no eval)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Filter runs
    runs_to_eval = RUNS
    if args.only:
        runs_to_eval = [r for r in RUNS if args.only.lower() in r[0].lower()]
        if not runs_to_eval:
            print(f"No runs matching '{args.only}'")
            sys.exit(1)

    if not args.table_only:
        # Verify all checkpoint files exist
        print(f"Verifying {len(runs_to_eval)} checkpoints...")
        missing = []
        for name, epochs, ckpt, algo in runs_to_eval:
            if not os.path.exists(ckpt):
                missing.append((name, ckpt))
                print(f"  MISSING: {name} -> {ckpt}")
        if missing:
            print(f"\n{len(missing)} checkpoints missing! Fix paths before running.")
            sys.exit(1)
        print(f"All {len(runs_to_eval)} checkpoints exist.\n")

        # Run evaluations
        total = len(runs_to_eval)
        successes = 0
        failures = 0
        skipped = 0
        total_time = 0.0

        for i, (name, epochs, ckpt, algo) in enumerate(runs_to_eval):
            out_path = os.path.join(RESULTS_DIR, f"{name}.json")

            if args.skip_existing and os.path.exists(out_path):
                print(f"[{i+1}/{total}] SKIP (exists): {name}")
                skipped += 1
                continue

            print(f"\n[{i+1}/{total}] Evaluating: {name} (epoch {epochs}, algo={algo})")
            success, _, elapsed = run_eval(name, ckpt, algo, RESULTS_DIR, args.dry_run)
            total_time += elapsed

            if success:
                successes += 1
            else:
                failures += 1

            # Progress estimate
            done = successes + failures
            if done > 0 and not args.dry_run:
                avg_time = total_time / done
                remaining = (total - done - skipped) * avg_time
                print(f"  Progress: {done}/{total-skipped} done, "
                      f"~{remaining/60:.0f} min remaining")

        if not args.dry_run:
            print(f"\n{'='*60}")
            print(f"BATCH COMPLETE: {successes} OK, {failures} FAILED, {skipped} SKIPPED")
            print(f"Total time: {total_time/60:.1f} minutes")
            print(f"{'='*60}")

    # Load all results and print table
    all_results = []
    for name, epochs, ckpt, algo in RUNS:
        json_path = os.path.join(RESULTS_DIR, f"{name}.json")
        data = load_results(json_path)
        if data and "metrics" in data:
            all_results.append({
                "name": name,
                "epochs": epochs,
                "algo": algo,
                "checkpoint": ckpt,
                "metrics": data["metrics"],
            })

    print_master_table(all_results)

    # Save master table as JSON too
    master_path = os.path.join(RESULTS_DIR, "_master_table.json")
    with open(master_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nMaster table JSON: {master_path}")


if __name__ == "__main__":
    main()
