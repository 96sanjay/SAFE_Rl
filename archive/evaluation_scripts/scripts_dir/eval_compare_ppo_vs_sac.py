#!/usr/bin/env python3
"""
Apple-to-apple evaluation: PPOLagMulti vs SACLagMulti at epoch 65.

Both models trained with identical env/reward/constraint configs from
run_r25b_report_stable.sh — only the algorithm differs.

Reuses env building, obs normalization, and metrics logic from eval_all_proper.py.
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np
import torch

PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
sys.path.insert(0, PROJECT)

# Import shared helpers from eval_all_proper
import eval_all_proper
from eval_all_proper import (
    SCHEMA_PATH,
    SEED,
    TOTAL_STEPS,
    EvalMetrics,
    build_env,
    evaluate_run,
    get_citylearn_env,
    log,
    normalize_obs,
    parse_exports,
    suppress_stdout,
)

# ---------------------------------------------------------------------------
# Patch get_action_names to walk deeper through CMDP wrapper chain
# ---------------------------------------------------------------------------
_original_get_action_names = eval_all_proper.get_action_names

def _patched_get_action_names(env) -> list:
    """Walk entire wrapper chain to find action_names (CMDP wrappers hide them)."""
    _e = env
    for _ in range(40):
        act_names_raw = getattr(_e, "action_names", None)
        if act_names_raw is not None:
            if isinstance(act_names_raw, list) and len(act_names_raw) > 0:
                if isinstance(act_names_raw[0], list):
                    return [n for sub in act_names_raw for n in sub]
                return list(act_names_raw)
        # Try all wrapper attributes
        nxt = None
        for attr in ("env", "base", "_env", "unwrapped", "raw_env"):
            candidate = getattr(_e, attr, None)
            if candidate is not None and candidate is not _e:
                nxt = candidate
                break
        if nxt is None:
            break
        _e = nxt
    return []

# Monkey-patch so evaluate_run uses our deeper walker
eval_all_proper.get_action_names = _patched_get_action_names

# ---------------------------------------------------------------------------
# Checkpoint paths
# ---------------------------------------------------------------------------
PPO_CKPT = os.path.join(
    PROJECT,
    "runs/r25b_report_stable/5bld/"
    "PPOLagMulti-{CityLearnSafety-V2G-v2}/"
    "seed-042-2026-03-27-04-04-26/torch_save/epoch-65.pt",
)
SAC_CKPT = os.path.join(
    os.path.expanduser("~"),
    "remote/runs/r25b_report_stable_sac_full_100ep_seed42/5bld/"
    "SACLagMulti-{CityLearnSafety-V2G-v2}/"
    "seed-042-2026-04-03-02-40-22/torch_save/epoch-65.pt",
)
SCRIPT_PATH = os.path.join(PROJECT, "run_r25b_report_stable.sh")

# ---------------------------------------------------------------------------
# Actor loaders
# ---------------------------------------------------------------------------

def load_ppo_actor(ckpt_path: str, obs_dim: int = 199, act_dim: int = 9):
    """Load PPO actor: Sequential(Linear→Tanh→Linear→Tanh→Linear), tanh output."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pi = ckpt["pi"]

    actor = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, 256),
        torch.nn.Tanh(),
        torch.nn.Linear(256, 256),
        torch.nn.Tanh(),
        torch.nn.Linear(256, act_dim),
    )
    key_map = {
        "mean.0.weight": "0.weight", "mean.0.bias": "0.bias",
        "mean.2.weight": "2.weight", "mean.2.bias": "2.bias",
        "mean.4.weight": "4.weight", "mean.4.bias": "4.bias",
    }
    sd = {seq_k: pi[omni_k] for omni_k, seq_k in key_map.items()}
    actor.load_state_dict(sd)
    actor.eval()

    norm_mean, norm_std, norm_clip = _load_normalizer(ckpt)
    return actor, norm_mean, norm_std, norm_clip


def load_sac_actor(ckpt_path: str, obs_dim: int = 199, act_dim: int = 9):
    """Load SAC actor: Sequential(Linear→ReLU→Linear→ReLU→Linear(→18)), tanh(mean[:9])."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pi = ckpt["pi"]

    # SAC outputs 18 dims: first 9 = mean, last 9 = log_std (we discard log_std)
    actor = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, act_dim * 2),  # 18
    )
    key_map = {
        "net.0.weight": "0.weight", "net.0.bias": "0.bias",
        "net.2.weight": "2.weight", "net.2.bias": "2.bias",
        "net.4.weight": "4.weight", "net.4.bias": "4.bias",
    }
    sd = {seq_k: pi[omni_k] for omni_k, seq_k in key_map.items()}
    actor.load_state_dict(sd)
    actor.eval()

    norm_mean, norm_std, norm_clip = _load_normalizer(ckpt)
    return actor, norm_mean, norm_std, norm_clip


def _load_normalizer(ckpt):
    nd = ckpt.get("obs_normalizer")
    if nd is None:
        return None, None, None
    mean = nd["_mean"].numpy()
    std = np.maximum(nd["_std"].numpy(), 1e-8)
    clip = nd["_clip"].numpy()
    return mean, std, clip


# ---------------------------------------------------------------------------
# Policy functions
# ---------------------------------------------------------------------------

def make_ppo_policy(actor, norm_mean, norm_std, norm_clip):
    def policy(obs):
        obs_n = normalize_obs(obs, norm_mean, norm_std, norm_clip)
        t = torch.as_tensor(obs_n, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = actor(t)
            return torch.tanh(out).squeeze(0).numpy()
    return policy


def make_sac_policy(actor, norm_mean, norm_std, norm_clip):
    def policy(obs):
        obs_n = normalize_obs(obs, norm_mean, norm_std, norm_clip)
        t = torch.as_tensor(obs_n, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = actor(t).squeeze(0)
            mean = out[:9]  # first 9 = mean, last 9 = log_std
            return torch.tanh(mean).numpy()
    return policy


# ---------------------------------------------------------------------------
# C2 fix: evaluate_run uses b.electrical_storage_devices (wrong — it's
# b.electrical_storage, a single Battery object). Recompute C2 properly.
# ---------------------------------------------------------------------------

def _get_deep_citylearn_env(env):
    """Walk to the deepest CityLearnEnv that has actual Battery objects."""
    e = env
    for _ in range(40):
        nxt = None
        for attr in ("env", "base", "_env", "unwrapped"):
            candidate = getattr(e, attr, None)
            if candidate is not None and candidate is not e:
                nxt = candidate
                break
        if nxt is None:
            break
        e = nxt
    return e  # should be CityLearnEnv


def recompute_c2(env, metrics: EvalMetrics):
    """Fix C2 by reading battery SoC from b.electrical_storage (not _devices)."""
    city = _get_deep_citylearn_env(env)
    if not hasattr(city, "buildings"):
        return
    c2_viol = 0
    c2_total = 0
    for b in city.buildings:
        es = getattr(b, "electrical_storage", None)
        if es is None:
            continue
        soc_arr = getattr(es, "soc", None)
        if soc_arr is None or not hasattr(soc_arr, "__len__"):
            continue
        n = min(len(soc_arr), metrics.steps_completed)
        for t in range(n):
            soc_val = float(soc_arr[t])
            c2_total += 1
            if soc_val > 0.95 or soc_val < 0.0:
                c2_viol += 1
    metrics.c2_violations = c2_viol
    metrics.c2_total = c2_total
    metrics.c2_violation_pct = 100.0 * c2_viol / max(c2_total, 1)


# ---------------------------------------------------------------------------
# Formatted output
# ---------------------------------------------------------------------------

def print_comparison(ppo: EvalMetrics, sac: EvalMetrics):
    """Print side-by-side comparison table."""

    def row(label, ppo_val, sac_val, fmt=".1f", unit=""):
        pv = f"{ppo_val:{fmt}}{unit}"
        sv = f"{sac_val:{fmt}}{unit}"
        log(f"  {label:<30} {pv:>14} {sv:>14}")

    def row_int(label, ppo_val, sac_val, unit=""):
        pv = f"{ppo_val}{unit}"
        sv = f"{sac_val}{unit}"
        log(f"  {label:<30} {pv:>14} {sv:>14}")

    log()
    log("=" * 62)
    log("  PPO vs SAC  —  Epoch 65  —  Apple-to-Apple Comparison")
    log("=" * 62)
    log(f"  {'Metric':<30} {'PPO':>14} {'SAC':>14}")
    log(f"  {'-'*30} {'-'*14} {'-'*14}")

    log()
    log("  --- Constraint Violations ---")
    row_int("C0 departures (total)", ppo.c0_total_departures, sac.c0_total_departures)
    row_int("C0 violated", ppo.c0_violated, sac.c0_violated)
    row("C0 violation %", ppo.c0_violation_pct, sac.c0_violation_pct, unit="%")
    row("C2 violation %", ppo.c2_violation_pct, sac.c2_violation_pct, unit="%")
    row("C3 violation %", ppo.c3_violation_pct, sac.c3_violation_pct, unit="%")
    row("C4 violation %", ppo.c4_violation_pct, sac.c4_violation_pct, unit="%")

    log()
    log("  --- EV Departure Diagnostics ---")
    row("Mean departure SoC", ppo.c0_mean_dep_soc, sac.c0_mean_dep_soc, fmt=".3f")
    row("Mean required SoC", ppo.c0_mean_req_soc, sac.c0_mean_req_soc, fmt=".3f")
    row("Mean deficit (violated)", ppo.c0_mean_deficit, sac.c0_mean_deficit, fmt=".3f")

    log()
    log("  --- V2G / EV Metrics ---")
    row("EV charge %", ppo.ev_charge_pct, sac.ev_charge_pct, unit="%")
    row("EV V2G discharge %", ppo.ev_v2g_pct, sac.ev_v2g_pct, unit="%")
    row("V2G during peak %", ppo.ev_v2g_peak_pct, sac.ev_v2g_peak_pct, unit="%")
    row("Total V2G energy (kWh)", ppo.ev_total_v2g_kwh, sac.ev_total_v2g_kwh, fmt=".0f", unit=" kWh")

    log()
    log("  --- Battery Cycling ---")
    row("Charge %", ppo.batt_charge_pct, sac.batt_charge_pct, unit="%")
    row("Discharge %", ppo.batt_discharge_pct, sac.batt_discharge_pct, unit="%")
    row("Idle %", ppo.batt_idle_pct, sac.batt_idle_pct, unit="%")
    row("Solar-hour avg action", ppo.batt_solar_avg_action, sac.batt_solar_avg_action, fmt=".3f")
    row("Peak-hour avg action", ppo.batt_peak_avg_action, sac.batt_peak_avg_action, fmt=".3f")
    row_int("Daily cycling days", ppo.batt_daily_cycling, sac.batt_daily_cycling)

    log()
    log("  --- CityLearn KPIs ---")
    row("Import (kWh)", ppo.total_import_kwh, sac.total_import_kwh, fmt=".0f")
    row("Export (kWh)", ppo.total_export_kwh, sac.total_export_kwh, fmt=".0f")
    row("Peak NEC (kW)", ppo.peak_nec_kw, sac.peak_nec_kw, fmt=".2f")
    row("Ramping (kWh)", ppo.ramping_kwh, sac.ramping_kwh, fmt=".0f")
    row("Electricity cost", ppo.electricity_cost, sac.electricity_cost, fmt=".0f")
    row("Load factor", ppo.load_factor, sac.load_factor, fmt=".3f")

    log()
    row_int("Steps completed", ppo.steps_completed, sac.steps_completed)
    row("Eval time (s)", ppo.eval_seconds, sac.eval_seconds, fmt=".1f")
    log("=" * 62)


def save_csv(ppo: EvalMetrics, sac: EvalMetrics, path: str):
    """Save results as CSV for downstream use."""
    fields = [
        "metric", "PPO", "SAC",
    ]
    rows = [
        ("c0_total_departures", ppo.c0_total_departures, sac.c0_total_departures),
        ("c0_violated", ppo.c0_violated, sac.c0_violated),
        ("c0_violation_pct", ppo.c0_violation_pct, sac.c0_violation_pct),
        ("c0_mean_dep_soc", ppo.c0_mean_dep_soc, sac.c0_mean_dep_soc),
        ("c0_mean_req_soc", ppo.c0_mean_req_soc, sac.c0_mean_req_soc),
        ("c0_mean_deficit", ppo.c0_mean_deficit, sac.c0_mean_deficit),
        ("c2_violation_pct", ppo.c2_violation_pct, sac.c2_violation_pct),
        ("c3_violations", ppo.c3_violations, sac.c3_violations),
        ("c3_total", ppo.c3_total, sac.c3_total),
        ("c3_violation_pct", ppo.c3_violation_pct, sac.c3_violation_pct),
        ("c4_violations", ppo.c4_violations, sac.c4_violations),
        ("c4_total", ppo.c4_total, sac.c4_total),
        ("c4_violation_pct", ppo.c4_violation_pct, sac.c4_violation_pct),
        ("ev_charge_pct", ppo.ev_charge_pct, sac.ev_charge_pct),
        ("ev_v2g_pct", ppo.ev_v2g_pct, sac.ev_v2g_pct),
        ("ev_v2g_peak_pct", ppo.ev_v2g_peak_pct, sac.ev_v2g_peak_pct),
        ("ev_total_v2g_kwh", ppo.ev_total_v2g_kwh, sac.ev_total_v2g_kwh),
        ("batt_charge_pct", ppo.batt_charge_pct, sac.batt_charge_pct),
        ("batt_discharge_pct", ppo.batt_discharge_pct, sac.batt_discharge_pct),
        ("batt_idle_pct", ppo.batt_idle_pct, sac.batt_idle_pct),
        ("batt_solar_avg_action", ppo.batt_solar_avg_action, sac.batt_solar_avg_action),
        ("batt_peak_avg_action", ppo.batt_peak_avg_action, sac.batt_peak_avg_action),
        ("batt_daily_cycling", ppo.batt_daily_cycling, sac.batt_daily_cycling),
        ("total_import_kwh", ppo.total_import_kwh, sac.total_import_kwh),
        ("total_export_kwh", ppo.total_export_kwh, sac.total_export_kwh),
        ("peak_nec_kw", ppo.peak_nec_kw, sac.peak_nec_kw),
        ("ramping_kwh", ppo.ramping_kwh, sac.ramping_kwh),
        ("electricity_cost", ppo.electricity_cost, sac.electricity_cost),
        ("load_factor", ppo.load_factor, sac.load_factor),
        ("steps_completed", ppo.steps_completed, sac.steps_completed),
        ("eval_seconds", ppo.eval_seconds, sac.eval_seconds),
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            w.writerow(r)
    log(f"CSV saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log("=" * 62)
    log("  PPO vs SAC — Apple-to-Apple Evaluation at Epoch 65")
    log(f"  Schema: {SCHEMA_PATH}")
    log(f"  Seed: {SEED}, Steps: {TOTAL_STEPS}")
    log("=" * 62)
    log()

    # Verify checkpoints exist
    for name, path in [("PPO", PPO_CKPT), ("SAC", SAC_CKPT)]:
        if not os.path.exists(path):
            log(f"ERROR: {name} checkpoint not found: {path}")
            sys.exit(1)
        log(f"  {name}: {path}")
    log()

    # Parse env vars (identical for both)
    env_vars = parse_exports(SCRIPT_PATH)
    metadata = {
        "saute": env_vars.get("CITYLEARN_EV_SAUTE", "0") == "1",
        "action_mask": False,
        "batt_clamp": False,
        "bc_warmstart": False,
        "pid_lagrange": env_vars.get("CITYLEARN_PID_LAGRANGE", "0") == "1",
        "curriculum": True,
    }

    # --- Evaluate PPO ---
    log(">>> Evaluating PPO (epoch 65) ...")
    with suppress_stdout():
        env_ppo = build_env(env_vars)
    obs_dim = env_ppo.observation_space.shape[0]
    act_dim = env_ppo.action_space.shape[0]
    log(f"  Env: obs_dim={obs_dim}, act_dim={act_dim}")

    actor_ppo, nm, ns, nc = load_ppo_actor(PPO_CKPT, obs_dim, act_dim)
    log(f"  PPO actor loaded, normalizer={'YES' if nm is not None else 'NO'}")
    policy_ppo = make_ppo_policy(actor_ppo, nm, ns, nc)
    result_ppo = evaluate_run("PPO_ep65", env_ppo, policy_ppo, env_vars, metadata)
    recompute_c2(env_ppo, result_ppo)
    log(f"  C0: {result_ppo.c0_violated}/{result_ppo.c0_total_departures} "
        f"({result_ppo.c0_violation_pct:.1f}%), "
        f"C3: {result_ppo.c3_violation_pct:.1f}%, C4: {result_ppo.c4_violation_pct:.1f}%")
    log()

    # --- Evaluate SAC ---
    log(">>> Evaluating SAC (epoch 65) ...")
    with suppress_stdout():
        env_sac = build_env(env_vars)
    obs_dim_s = env_sac.observation_space.shape[0]
    act_dim_s = env_sac.action_space.shape[0]
    log(f"  Env: obs_dim={obs_dim_s}, act_dim={act_dim_s}")

    actor_sac, nm_s, ns_s, nc_s = load_sac_actor(SAC_CKPT, obs_dim_s, act_dim_s)
    log(f"  SAC actor loaded, normalizer={'YES' if nm_s is not None else 'NO'}")
    policy_sac = make_sac_policy(actor_sac, nm_s, ns_s, nc_s)
    result_sac = evaluate_run("SAC_ep65", env_sac, policy_sac, env_vars, metadata)
    recompute_c2(env_sac, result_sac)
    log(f"  C0: {result_sac.c0_violated}/{result_sac.c0_total_departures} "
        f"({result_sac.c0_violation_pct:.1f}%), "
        f"C3: {result_sac.c3_violation_pct:.1f}%, C4: {result_sac.c4_violation_pct:.1f}%")
    log()

    # --- Print comparison ---
    print_comparison(result_ppo, result_sac)

    # --- Save CSV ---
    csv_path = os.path.join(PROJECT, "docs", "ppo_vs_sac_epoch65.csv")
    save_csv(result_ppo, result_sac, csv_path)

    log("\nDone.")


if __name__ == "__main__":
    main()
