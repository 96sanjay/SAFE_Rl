
#!/usr/bin/env python3
"""
Compare KPI CSVs: Agent vs RBC (both produced via CityLearnSafetyEnv logger).

Outputs:
- Prints a side-by-side report in terminal
- Writes one summary CSV to: evaluation_pipeline/processed_tables/<out_name>.csv

Usage:
  python evaluation_pipeline/scripts/compare_kpis_agent_vs_rbc.py \
    --agent_csv runs/kpi_logs/Evaluated_PPOLag_bill_evscale2_ep35_seed000.csv \
    --rbc_csv   runs/kpi_logs/RBC_Greedy_V3.csv \
    --out_name  cmp_agent_vs_rbc_bill_ep35_seed000
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def safe_float_series(df, col):
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def last_row_kpis(df):
    """Return dict of citylearn_* KPIs from last row (single episode-end values)."""
    out = {}
    last = df.tail(1)
    for c in df.columns:
        if c.startswith("citylearn_"):
            out[c] = float(last[c].iloc[0]) if c in last.columns else np.nan
    return out


def compute_step_metrics(df):
    """Compute robust step-level metrics from the full KPI CSV."""
    dep = safe_float_series(df, "ev_departure_departures").fillna(0.0) > 0
    deficit = safe_float_series(df, "ev_departure_deficit_kwh").fillna(0.0)
    avoid = safe_float_series(df, "ev_avoidable_deficit_kwh").fillna(0.0)
    unavoid = safe_float_series(df, "ev_unavoidable_deficit_kwh").fillna(0.0)
    cost = safe_float_series(df, "cost").fillna(0.0)

    dep_steps = int(dep.sum())
    miss_steps = int(((dep) & (deficit > 0)).sum())

    # EV action behavior (if present)
    action_ev_cols = sorted(
        [c for c in df.columns if re.fullmatch(r"action_ev_\d+", c)],
        key=lambda x: int(x.split("_")[-1]),
    )
    if action_ev_cols:
        ev = df[action_ev_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        ev_mean = ev.mean(axis=1)
        ev_pos_frac = float((ev_mean > 1e-3).mean())
        ev_neg_frac = float((ev_mean < -1e-3).mean())
        ev_mean_mean = float(ev_mean.mean())
    else:
        ev_pos_frac = np.nan
        ev_neg_frac = np.nan
        ev_mean_mean = np.nan

    # Reward / Bill (if present)
    reward = safe_float_series(df, "reward")
    bill = safe_float_series(df, "reward_bill_raw")
    step_cost = safe_float_series(df, "step_cost")
    import_kwh = safe_float_series(df, "grid_import_kwh")
    export_kwh = safe_float_series(df, "grid_export_kwh")

    out = {
        "rows": int(len(df)),
        "episodes": int(df["episode"].nunique()) if "episode" in df.columns else np.nan,
        "dep_steps": dep_steps,
        "miss_steps": miss_steps,
        "miss_rate": float(miss_steps / max(1, dep_steps)),
        "total_deficit_kwh": float(deficit[dep].sum()),
        "total_avoidable_kwh": float(avoid[dep].sum()),
        "total_unavoidable_kwh": float(unavoid[dep].sum()),
        "deficit_per_dep": float(deficit[dep].sum() / max(1, dep_steps)),
        "avoidable_per_dep": float(avoid[dep].sum() / max(1, dep_steps)),
        "unavoidable_per_dep": float(unavoid[dep].sum() / max(1, dep_steps)),
        "total_cmdp_cost": float(cost.sum()),
        "mean_cost_per_step": float(cost.mean()),
        "cost_pos_steps": int((cost > 0).sum()),
        "cost_pos_frac": float((cost > 0).mean()),
        "ev_mean_action_mean": ev_mean_mean,
        "ev_pos_frac": ev_pos_frac,
        "ev_neg_frac": ev_neg_frac,
        "mean_reward": float(reward.mean()) if reward.notna().any() else np.nan,
        "mean_bill_raw": float(bill.mean()) if bill.notna().any() else np.nan,
        "sum_bill_raw": float(bill.sum()) if bill.notna().any() else np.nan,
        "sum_step_cost": float(step_cost.sum()) if step_cost.notna().any() else np.nan,
        "sum_import_kwh": float(import_kwh.sum()) if import_kwh.notna().any() else np.nan,
        "sum_export_kwh": float(export_kwh.sum()) if export_kwh.notna().any() else np.nan,
    }

    # Identity check (only on departure rows)
    if dep_steps > 0:
        ident_err = (deficit[dep] - (avoid[dep] + unavoid[dep])).abs().max()
        out["identity_max_abs_err"] = float(ident_err)
    else:
        out["identity_max_abs_err"] = np.nan

    # Missing action samples (if present)
    miss_actions = safe_float_series(df, "ev_missing_action_samples").fillna(0.0)
    out["missing_action_samples_total"] = float(miss_actions.sum())

    return out


def pretty_print_compare(title, A, B):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    keys = sorted(set(A.keys()) | set(B.keys()))
    for k in keys:
        a = A.get(k, np.nan)
        b = B.get(k, np.nan)
        if isinstance(a, float) and np.isnan(a) and isinstance(b, float) and np.isnan(b):
            continue
        try:
            diff = (a - b) if (np.isfinite(a) and np.isfinite(b)) else np.nan
        except Exception:
            diff = np.nan
        print(f"{k:35s}  agent={a:14.6f}  rbc={b:14.6f}  diff={diff:14.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent_csv", required=True)
    ap.add_argument("--rbc_csv", required=True)
    ap.add_argument("--out_name", default="cmp_agent_vs_rbc")
    args = ap.parse_args()

    agent_path = Path(args.agent_csv)
    rbc_path = Path(args.rbc_csv)
    assert agent_path.exists(), f"Agent CSV not found: {agent_path}"
    assert rbc_path.exists(), f"RBC CSV not found: {rbc_path}"

    dfA = pd.read_csv(agent_path)
    dfB = pd.read_csv(rbc_path)

    # Step-level metrics
    A_step = compute_step_metrics(dfA)
    B_step = compute_step_metrics(dfB)

    # Episode-end CityLearn KPIs (from last row)
    A_city = last_row_kpis(dfA)
    B_city = last_row_kpis(dfB)

    # Print comparisons
    pretty_print_compare("STEP-LEVEL METRICS (computed from whole CSV)", A_step, B_step)
    pretty_print_compare("CITYLEARN EPISODE-END KPIs (from last row)", A_city, B_city)

    # Write one tidy summary CSV
    out_dir = Path("evaluation_pipeline/processed_tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{args.out_name}.csv"

    # Build a single-row dataframe (agent/rbc/diff columns)
    rows = {}
    for k in sorted(set(A_step.keys()) | set(B_step.keys())):
        rows[f"{k}__agent"] = A_step.get(k, np.nan)
        rows[f"{k}__rbc"] = B_step.get(k, np.nan)
        try:
            rows[f"{k}__diff"] = rows[f"{k}__agent"] - rows[f"{k}__rbc"]
        except Exception:
            rows[f"{k}__diff"] = np.nan

    for k in sorted(set(A_city.keys()) | set(B_city.keys())):
        rows[f"{k}__agent"] = A_city.get(k, np.nan)
        rows[f"{k}__rbc"] = B_city.get(k, np.nan)
        try:
            rows[f"{k}__diff"] = rows[f"{k}__agent"] - rows[f"{k}__rbc"]
        except Exception:
            rows[f"{k}__diff"] = np.nan

    out_df = pd.DataFrame([rows])
    out_df.to_csv(out_csv, index=False)

    print("\n" + "=" * 90)
    print("WROTE SUMMARY CSV")
    print("=" * 90)
    print(str(out_csv))


if __name__ == "__main__":
    main()
