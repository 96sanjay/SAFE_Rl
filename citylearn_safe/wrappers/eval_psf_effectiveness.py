
#!/usr/bin/env python3
"""
PSF Effectiveness Evaluation: compare constraint violations WITH vs WITHOUT PSF.

Runs a full episode (8760 steps = 1 year hourly) using random actions,
measuring all 4 constraint families:
  1. EV departure deficit (SoC at departure vs required)
  2. Battery SoC bounds [soc_low, soc_high]
  3. Per-building power capacity
  4. District grid import

Usage:
    cd /home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork
    conda activate citylearn
    python3 scripts/eval_psf_effectiveness.py

Output: side-by-side comparison table + CSV export.
"""

from __future__ import annotations

import os
import sys
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

SCHEMA_PATH = os.environ.get(
    "CITYLEARN_SCHEMA",
    os.path.join(PROJECT_ROOT, "data",
                 "citylearn_challenge_2022_phase_all_plus_evs", "schema.json"),
)


def normalize_schema(path: str) -> dict:
    with open(path) as f:
        schema = json.load(f)
    base = os.path.dirname(path)
    root = schema.get("root_directory", "")
    if not root or root in (".", "./"):
        root = base
    elif not os.path.isabs(root):
        root = os.path.normpath(os.path.join(base, root))
    schema["root_directory"] = root
    return schema


# ---------------------------------------------------------------------------
# Violation tracker
# ---------------------------------------------------------------------------
@dataclass
class ViolationTracker:
    """Track per-step violations for all 4 constraint families."""
    name: str
    total_steps: int = 0
    total_reward: float = 0.0

    # EV departure
    ev_departure_events: int = 0
    ev_deficit_total_soc: float = 0.0
    ev_deficit_steps: int = 0  # steps with any EV departure deficit

    # Battery SoC bounds (per-building violations)
    battery_soc_violation_steps: int = 0       # steps with any building violating
    battery_soc_violation_building_steps: int = 0  # sum of (buildings violating) across steps
    battery_soc_total_buildings_checked: int = 0

    # Building power capacity
    building_power_violation_steps: int = 0
    building_power_violation_building_steps: int = 0

    # Grid import
    grid_import_violation_steps: int = 0
    grid_import_excess_total_kwh: float = 0.0

    # PSF stats
    psf_interventions: int = 0
    psf_ev_interventions: int = 0
    psf_battery_interventions: int = 0
    psf_grid_interventions: int = 0

    def record_step(self, info: Dict, n_buildings: int = 17):
        self.total_steps += 1
        self.total_reward += float(info.get("reward", 0.0))

        # EV departure
        ev_deps = int(info.get("ev_departure_departures", 0))
        ev_deficit = float(info.get("ev_departure_deficit_kwh", 0.0))
        if ev_deps > 0:
            self.ev_departure_events += ev_deps
        if ev_deficit > 1e-6:
            self.ev_deficit_total_soc += ev_deficit
            self.ev_deficit_steps += 1

        # Battery SoC: use env's own violation computation (correct thresholds)
        batt_viol_flag = float(info.get("battery_soc_violation", 0.0))
        batt_viol_cnt = float(info.get("battery_soc_violation_count", 0.0))
        self.battery_soc_total_buildings_checked += n_buildings
        if batt_viol_flag > 0.5:
            self.battery_soc_violation_steps += 1
        self.battery_soc_violation_building_steps += int(batt_viol_cnt)

        # Building power
        bld_viol = float(info.get("building_power_violation", 0.0))
        bld_cnt = float(info.get("building_power_violation_count", 0.0))
        if bld_viol > 0.5:
            self.building_power_violation_steps += 1
        self.building_power_violation_building_steps += int(bld_cnt)

        # Grid import
        grid_viol = float(info.get("grid_power_violation", 0.0))
        grid_excess = float(info.get("cost_stems_grid_power", 0.0))
        if grid_viol > 0.5:
            self.grid_import_violation_steps += 1
            self.grid_import_excess_total_kwh += grid_excess

        # PSF
        if info.get("psf_any_intervention", False):
            self.psf_interventions += 1
        self.psf_ev_interventions += int(info.get("psf_ev_interventions", 0))
        self.psf_battery_interventions += int(info.get("psf_battery_interventions", 0))
        self.psf_grid_interventions += int(info.get("psf_grid_interventions", 0))

    def summary(self) -> Dict[str, float]:
        T = max(1, self.total_steps)
        n_bld = 17
        return {
            "steps": T,
            "avg_reward": self.total_reward / T,
            "total_reward": self.total_reward,
            # EV
            "ev_departures": self.ev_departure_events,
            "ev_deficit_steps": self.ev_deficit_steps,
            "ev_deficit_rate_%": 100.0 * self.ev_deficit_steps / T,
            "ev_deficit_total_kwh": self.ev_deficit_total_soc,
            # Battery SoC
            "batt_viol_steps": self.battery_soc_violation_steps,
            "batt_viol_rate_%": 100.0 * self.battery_soc_violation_steps / T,
            "batt_viol_building_rate_%": (
                100.0 * self.battery_soc_violation_building_steps
                / max(1, self.battery_soc_total_buildings_checked)
            ),
            # Building power
            "bld_power_viol_steps": self.building_power_violation_steps,
            "bld_power_viol_rate_%": 100.0 * self.building_power_violation_steps / T,
            # Grid import
            "grid_viol_steps": self.grid_import_violation_steps,
            "grid_viol_rate_%": 100.0 * self.grid_import_violation_steps / T,
            "grid_excess_total_kwh": self.grid_import_excess_total_kwh,
            # PSF
            "psf_interventions": self.psf_interventions,
            "psf_intervention_rate_%": 100.0 * self.psf_interventions / T,
            "psf_ev_interventions": self.psf_ev_interventions,
            "psf_battery_interventions": self.psf_battery_interventions,
            "psf_grid_interventions": self.psf_grid_interventions,
        }


# ---------------------------------------------------------------------------
# Rollout function
# ---------------------------------------------------------------------------
def run_rollout(
    use_psf: bool,
    psf_kwargs: Optional[Dict] = None,
    seed: int = 42,
    max_steps: int = 8760,
    action_mode: str = "random",
) -> ViolationTracker:
    """
    Run one full episode, return violation tracker.

    action_mode:
        "random" - uniform random from action_space
        "zero"   - all-zeros (do nothing)
        "full_charge" - all +1.0 (max charge everything)
    """
    from citylearn.citylearn import CityLearnEnv
    from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
    from citylearn_safe.psf import PredictiveSafetyFilterWrapper

    schema = normalize_schema(SCHEMA_PATH)

    # Suppress KPI logging noise
    run_name = f"psf_eval_{'psf' if use_psf else 'nopsf'}_{action_mode}"
    os.environ["CITYLEARN_KPI_RUN_NAME"] = run_name

    base_env = CityLearnEnv(schema=schema)
    safety_env = CityLearnSafetyEnvV3(base_env)

    # Init RBC if needed
    rbc_agent = None
    rbc_agent_dims = None
    if action_mode == "rbc":
        from evaluation.agents.rbc import IntelligentRBC
        # RBC needs flat action_names + correct action_dim.
        # Multi-agent env has nested names [[...],[...]] and list-of-Box spaces.
        # Create a thin proxy that flattens these for RBC.
        raw_city = base_env
        cur = safety_env
        for _ in range(20):
            blds = getattr(cur, "buildings", None)
            ts = getattr(cur, "time_step", None)
            if blds is not None and hasattr(blds, "__len__") and len(blds) > 0 and ts is not None:
                raw_city = cur
                break
            cur = getattr(cur, "env", None)
            if cur is None:
                break

        # Flatten nested action_names
        raw_names = getattr(raw_city, "action_names", [])
        if isinstance(raw_names, list) and len(raw_names) > 0 and isinstance(raw_names[0], list):
            flat_names = [n for sub in raw_names for n in sub]
        else:
            flat_names = list(raw_names)

        class _FlatEnvView:
            """Proxy giving RBC flat action_names and scalar action_space.shape"""
            def __init__(self, raw, names):
                self._raw = raw
                self.action_names = names
                self.action_space = type("_S", (), {"shape": (len(names),)})()
            def __getattr__(self, name):
                return getattr(self._raw, name)

        rbc_agent = IntelligentRBC(_FlatEnvView(raw_city, flat_names), ev_mode="greedy")

        # Capture multi-agent dims for reshaping
        if isinstance(safety_env.action_space, list):
            rbc_agent_dims = [sp.shape[0] for sp in safety_env.action_space]

    if use_psf:
        kwargs = dict(
            horizon=24,
            correction_mode="heuristic",
            ev_urgency_threshold=0.5,
            verbose=0,
        )
        if psf_kwargs:
            kwargs.update(psf_kwargs)
        env = PredictiveSafetyFilterWrapper(safety_env, **kwargs)
        tracker = ViolationTracker(name=f"PSF ({action_mode})")
    else:
        env = safety_env
        tracker = ViolationTracker(name=f"No PSF ({action_mode})")

    is_multi = isinstance(env.action_space, list)
    obs, info = env.reset(seed=seed)

    for step_i in range(max_steps):
        if action_mode == "random":
            if is_multi:
                action = [sp.sample() for sp in env.action_space]
            else:
                action = env.action_space.sample()
        elif action_mode == "zero":
            if is_multi:
                action = [np.zeros(sp.shape, dtype=np.float32) for sp in env.action_space]
            else:
                action = np.zeros(env.action_space.shape, dtype=np.float32)
        elif action_mode == "full_charge":
            if is_multi:
                action = [np.ones(sp.shape, dtype=np.float32) for sp in env.action_space]
            else:
                action = np.ones(env.action_space.shape, dtype=np.float32)
        elif action_mode == "rbc":
            if rbc_agent is None:
                raise ValueError("RBC agent not initialized")
            flat_action = rbc_agent.predict(obs)
            # Reshape flat RBC output into list-of-arrays for multi-agent env
            if is_multi and rbc_agent_dims is not None:
                action = []
                offset = 0
                for d in rbc_agent_dims:
                    action.append(flat_action[offset:offset+d].copy())
                    offset += d
            elif is_multi:
                action = [flat_action]
            else:
                action = flat_action
        else:
            raise ValueError(f"Unknown action_mode: {action_mode}")

        obs, reward, term, trunc, info = env.step(action)
        tracker.record_step(info)

        if term or trunc:
            break

        # Progress
        if (step_i + 1) % 2000 == 0:
            pct = 100.0 * (step_i + 1) / max_steps
            print(f"  [{tracker.name}] {step_i+1}/{max_steps} ({pct:.0f}%)")

    return tracker


# ---------------------------------------------------------------------------
# Pretty print comparison
# ---------------------------------------------------------------------------
def print_comparison(a: ViolationTracker, b: ViolationTracker):
    sa = a.summary()
    sb = b.summary()

    def fmt(v, pct=False):
        if pct:
            return f"{v:8.2f}%"
        if isinstance(v, float):
            return f"{v:10.2f}"
        return f"{v:10d}"

    def delta_str(va, vb, lower_better=True):
        diff = vb - va
        if abs(diff) < 1e-6:
            return "    same"
        arrow = "↓" if (diff < 0 and lower_better) or (diff > 0 and not lower_better) else "↑"
        sign = "+" if diff > 0 else ""
        if isinstance(va, float):
            return f"  {arrow} {sign}{diff:.2f}"
        return f"  {arrow} {sign}{int(diff)}"

    print("\n" + "=" * 80)
    print(f"  COMPARISON: {a.name}  vs  {b.name}")
    print("=" * 80)

    header = f"{'Metric':<40} {'No PSF':>12} {'With PSF':>12} {'Delta':>12}"
    print(header)
    print("-" * 80)

    rows = [
        ("Steps", "steps", False, False),
        ("Avg Reward (per step)", "avg_reward", False, False),
        ("Total Reward", "total_reward", False, False),
        ("", None, None, None),
        ("── EV DEPARTURE ──", None, None, None),
        ("EV departures (count)", "ev_departures", False, False),
        ("Steps with EV deficit", "ev_deficit_steps", False, True),
        ("EV deficit rate", "ev_deficit_rate_%", True, True),
        ("EV deficit total (kWh)", "ev_deficit_total_kwh", False, True),
        ("", None, None, None),
        ("── BATTERY SoC BOUNDS ──", None, None, None),
        ("Steps with batt violation", "batt_viol_steps", False, True),
        ("Battery violation rate", "batt_viol_rate_%", True, True),
        ("Per-building violation rate", "batt_viol_building_rate_%", True, True),
        ("", None, None, None),
        ("── BUILDING POWER ──", None, None, None),
        ("Steps with bld pwr violation", "bld_power_viol_steps", False, True),
        ("Building power viol rate", "bld_power_viol_rate_%", True, True),
        ("", None, None, None),
        ("── GRID IMPORT ──", None, None, None),
        ("Steps with grid violation", "grid_viol_steps", False, True),
        ("Grid import viol rate", "grid_viol_rate_%", True, True),
        ("Grid excess total (kWh)", "grid_excess_total_kwh", False, True),
        ("", None, None, None),
        ("── PSF STATS ──", None, None, None),
        ("PSF interventions", "psf_interventions", False, False),
        ("PSF intervention rate", "psf_intervention_rate_%", True, False),
        ("PSF EV interventions", "psf_ev_interventions", False, False),
        ("PSF battery interventions", "psf_battery_interventions", False, False),
        ("PSF grid interventions", "psf_grid_interventions", False, False),
    ]

    for label, key, is_pct, lower_better in rows:
        if key is None:
            print(f"  {label}")
            continue

        va = sa.get(key, 0)
        vb = sb.get(key, 0)
        col_a = fmt(va, is_pct) if a.name.startswith("No") else fmt(vb, is_pct)
        col_b = fmt(vb, is_pct) if a.name.startswith("No") else fmt(va, is_pct)

        # Always: col_a = no_psf, col_b = psf
        va_nopsf = sa.get(key, 0) if "No" in a.name else sb.get(key, 0)
        vb_psf = sb.get(key, 0) if "No" in a.name else sa.get(key, 0)

        d = delta_str(va_nopsf, vb_psf, lower_better) if lower_better is not None else ""

        print(f"  {label:<38} {fmt(va_nopsf, is_pct):>12} {fmt(vb_psf, is_pct):>12} {d:>12}")

    print("=" * 80)

    # Quick verdict
    nopsf_s = sa if "No" in a.name else sb
    psf_s = sb if "No" in a.name else sa

    improved = 0
    total_checks = 0
    # Only check PSF-enforceable constraints (EV + battery)
    # Building power & grid are cost signals for RL, not PSF-enforceable
    psf_keys = ["ev_deficit_rate_%", "batt_viol_building_rate_%"]
    cost_keys = ["bld_power_viol_rate_%", "grid_viol_rate_%"]
    for key in psf_keys + cost_keys:
        total_checks += 1
        if psf_s[key] < nopsf_s[key] - 0.1:
            improved += 1

    target_met = 0
    for key in psf_keys + cost_keys:
        if psf_s[key] < 5.0:
            target_met += 1

    print(f"\n  Verdict: PSF improved {improved}/{total_checks} constraint families")
    print(f"  Target (<5% each): {target_met}/4 families meet target with PSF")

    if target_met == 4:
        print("  🎯 All 4 constraint families under 5% — target MET!")
    elif improved >= 2:
        print("  📈 Partial improvement — PSF is helping but may need tuning")
    else:
        print("  ⚠️  Limited improvement — check PSF parameters or action_mode")

    print()


# ---------------------------------------------------------------------------
# Export to CSV
# ---------------------------------------------------------------------------
def export_csv(trackers: List[ViolationTracker], path: str):
    import csv
    summaries = [t.summary() for t in trackers]
    all_keys = list(summaries[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name"] + all_keys)
        for t, s in zip(trackers, summaries):
            w.writerow([t.name] + [s[k] for k in all_keys])
    print(f"  Exported to: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="PSF Effectiveness Evaluation")
    parser.add_argument("--max-steps", type=int, default=8760,
                        help="Max steps per rollout (default: 8760 = 1 year)")
    parser.add_argument("--action-mode", type=str, default="random",
                        choices=["random", "zero", "full_charge", "rbc"],
                        help="Action generation mode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--urgency", type=float, default=0.5,
                        help="EV urgency threshold for PSF")
    parser.add_argument("--csv", type=str, default=None,
                        help="Export CSV path (optional)")
    args = parser.parse_args()

    print("=" * 60)
    print("PSF EFFECTIVENESS EVALUATION")
    print(f"  action_mode={args.action_mode}  max_steps={args.max_steps}")
    print(f"  seed={args.seed}  urgency_threshold={args.urgency}")
    print("=" * 60)

    # Run WITHOUT PSF
    print(f"\n[1/2] Running WITHOUT PSF ({args.action_mode} actions)...")
    t0 = time.time()
    tracker_nopsf = run_rollout(
        use_psf=False,
        seed=args.seed,
        max_steps=args.max_steps,
        action_mode=args.action_mode,
    )
    t1 = time.time()
    print(f"  Done in {t1-t0:.1f}s ({tracker_nopsf.total_steps} steps)")

    # Run WITH PSF
    print(f"\n[2/2] Running WITH PSF ({args.action_mode} actions)...")
    t0 = time.time()
    tracker_psf = run_rollout(
        use_psf=True,
        psf_kwargs={"ev_urgency_threshold": args.urgency},
        seed=args.seed,
        max_steps=args.max_steps,
        action_mode=args.action_mode,
    )
    t1 = time.time()
    print(f"  Done in {t1-t0:.1f}s ({tracker_psf.total_steps} steps)")

    # Compare
    print_comparison(tracker_nopsf, tracker_psf)

    # Export
    csv_path = args.csv or os.path.join(
        "runs", "kpi_logs", f"psf_eval_{args.action_mode}_{args.seed}.csv"
    )
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    export_csv([tracker_nopsf, tracker_psf], csv_path)


if __name__ == "__main__":
    main()