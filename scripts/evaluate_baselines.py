
"""
Complete Baseline Evaluation - Full Episode Analysis

Computes all metrics over 1 complete episode:
- Total CMDP cost breakdown
- EV departure shortfall (total, avoidable, unavoidable)
- Battery SoC violations (count, magnitude, hours)
- Economic/environmental metrics
- CityLearn official evaluation

Output format compatible with Safe-RL comparison.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


def _safe_pct(part: float, whole: float) -> float:
    """Return 100*part/whole, safely handling whole==0."""
    return 100.0 * part / whole if whole and whole > 0 else 0.0


def evaluate_baseline(kpi_csv_path: str, agent_name: str, step_hours: float = 1.0):
    """
    Complete evaluation of a baseline run.

    Args:
        kpi_csv_path: Path to KPI CSV file (e.g., 'runs/baselines/no_control/kpis.csv')
        agent_name: Name of agent (e.g., 'No-Control', 'Intelligent-RBC')
        step_hours: Duration of one environment step in hours (CityLearn is often 1.0)

    Returns:
        dict: Complete evaluation metrics
    """

    df = pd.read_csv(kpi_csv_path)

    print(f"\n{'='*60}")
    print(f"EVALUATING: {agent_name}")
    print(f"{'='*60}")
    print(f"Total steps: {len(df)}")

    # ========================================
    # 1. CMDP COST BREAKDOWN
    # ========================================
    total_cost = df['cost'].sum()
    building_soc_cost = df['cost_building_soc'].sum()
    ev_departure_cost = df['cost_ev_departure'].sum()

    avg_cost_per_step = total_cost / len(df) if len(df) > 0 else 0.0

    print(f"\n1. CMDP COST BREAKDOWN:")
    print(f"   Total CMDP cost:           {total_cost:.2f}")
    print(f"   - Building SoC violations: {building_soc_cost:.2f} ({_safe_pct(building_soc_cost, total_cost):.1f}%)")
    print(f"   - EV departure shortfall:  {ev_departure_cost:.2f} ({_safe_pct(ev_departure_cost, total_cost):.1f}%)")
    print(f"   Average cost per step:     {avg_cost_per_step:.4f}")

    # ========================================
    # 2. EV DEPARTURE SHORTFALL (Energy-wise Split)
    # ========================================
    ev_total_kwh = df['ev_departure_deficit_kwh'].sum()
    ev_avoidable_kwh = df['ev_avoidable_deficit_kwh'].sum()
    ev_unavoidable_kwh = df['ev_unavoidable_deficit_kwh'].sum()

    ev_departures_total = df['ev_departure_deficit_kwh'].notna().sum()
    ev_failures = (df['ev_departure_deficit_kwh'] > 0.001).sum()

    print(f"\n2. EV DEPARTURE SHORTFALL (Energy-wise):")
    print(f"   Total deficit:             {ev_total_kwh:.2f} kWh")
    print(f"   - Avoidable (policy fault):{ev_avoidable_kwh:.2f} kWh ({_safe_pct(ev_avoidable_kwh, ev_total_kwh):.1f}%)")
    print(f"   - Unavoidable (physics):   {ev_unavoidable_kwh:.2f} kWh ({_safe_pct(ev_unavoidable_kwh, ev_total_kwh):.1f}%)")
    print(f"   Departure events with deficit: {ev_failures}")

    # ========================================
    # 3. BATTERY SOC VIOLATIONS (Same Logic as Cost Function)
    # ========================================
    # Count violations (steps where soc_max > 0.95)
    soc_violation_count = (df['soc_max'] > 0.95).sum()
    soc_violation_pct = 100 * soc_violation_count / len(df) if len(df) > 0 else 0.0

    # Magnitude (total excess energy)
    battery_abuse_total_kwh = df['battery_abuse_kwh'].sum()

    # OPTION A: compute hours from battery_abuse_kwh
    # If your timestep isn't 1 hour, pass step_hours accordingly.
    abuse_steps = (df['battery_abuse_kwh'] > 0.0).sum()
    battery_abuse_hours = abuse_steps * step_hours

    # Per-step average when violated
    violated_steps = df[df['soc_max'] > 0.95]
    avg_soc_when_violated = violated_steps['soc_max'].mean() if len(violated_steps) > 0 else 0.0

    print(f"\n3. BATTERY SOC VIOLATIONS (SoC > 0.95):")
    print(f"   Violation count:           {soc_violation_count} steps ({soc_violation_pct:.1f}%)")
    print(f"   Total excess energy:       {battery_abuse_total_kwh:.2f} kWh")
    print(f"   Hours with any violation:  {battery_abuse_hours:.2f}")
    print(f"   Avg SoC when violated:     {avg_soc_when_violated:.4f}")
    print(f"   Max SoC observed:          {df['soc_max'].max():.4f}")

    # ========================================
    # 4. ECONOMIC METRICS
    # ========================================
    total_electricity_cost = df['step_cost'].sum() if 'step_cost' in df.columns else 0.0
    total_grid_import = df['grid_import_kwh'].sum() if 'grid_import_kwh' in df.columns else 0.0
    total_grid_export = df['grid_export_kwh'].sum() if 'grid_export_kwh' in df.columns else 0.0

    print(f"\n4. ECONOMIC METRICS:")
    print(f"   Total electricity cost:    ${total_electricity_cost:.2f}")
    print(f"   Total grid import:         {total_grid_import:.2f} kWh")
    print(f"   Total grid export:         {total_grid_export:.2f} kWh")
    print(f"   Net grid consumption:      {total_grid_import - total_grid_export:.2f} kWh")

    # ========================================
    # 5. ENVIRONMENTAL METRICS
    # ========================================
    total_carbon = 0.0
    if 'step_carbon_kg' in df.columns:
        total_carbon = df['step_carbon_kg'].sum()
        print(f"\n5. ENVIRONMENTAL METRICS:")
        print(f"   Total carbon emissions:    {total_carbon:.2f} kg CO2")

    # ========================================
    # 6. SUMMARY STATISTICS
    # ========================================
    total_reward = df['reward'].sum()
    avg_reward = df['reward'].mean()
    avg_soc = df['soc_mean'].mean() if 'soc_mean' in df.columns else 0.0

    print(f"\n6. SUMMARY STATISTICS:")
    print(f"   Total reward:              {total_reward:.2f}")
    print(f"   Average reward per step:   {avg_reward:.4f}")
    print(f"   Average SoC:               {avg_soc:.4f}")

    # ========================================
    # COMPILE RESULTS
    # ========================================
    results = {
        "agent": agent_name,
        "total_steps": len(df),

        # CMDP Cost
        "cmdp_cost_total": float(total_cost),
        "cmdp_cost_building_soc": float(building_soc_cost),
        "cmdp_cost_ev_departure": float(ev_departure_cost),
        "cmdp_cost_per_step": float(avg_cost_per_step),

        # EV Shortfall (Energy-wise)
        "ev_deficit_total_kwh": float(ev_total_kwh),
        "ev_deficit_avoidable_kwh": float(ev_avoidable_kwh),
        "ev_deficit_unavoidable_kwh": float(ev_unavoidable_kwh),
        "ev_deficit_avoidable_pct": float(_safe_pct(ev_avoidable_kwh, ev_total_kwh)),
        "ev_failures_count": int(ev_failures),

        # Battery SoC Violations
        "soc_violation_count": int(soc_violation_count),
        "soc_violation_pct": float(soc_violation_pct),
        "battery_abuse_total_kwh": float(battery_abuse_total_kwh),
        "battery_abuse_hours": float(battery_abuse_hours),
        "soc_max_observed": float(df['soc_max'].max()),

        # Economic
        "electricity_cost_total": float(total_electricity_cost),
        "grid_import_total_kwh": float(total_grid_import),
        "grid_export_total_kwh": float(total_grid_export),
        "net_grid_consumption_kwh": float(total_grid_import - total_grid_export),

        # Environmental
        "carbon_emissions_kg": float(total_carbon) if 'step_carbon_kg' in df.columns else 0.0,

        # Reward
        "total_reward": float(total_reward),
        "avg_reward_per_step": float(avg_reward),
        "avg_soc": float(avg_soc),
    }

    return results


def compare_agents(results_list):
    """
    Compare multiple agents side-by-side.

    Args:
        results_list: List of result dicts from evaluate_baseline()
    """

    print(f"\n{'='*80}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*80}")

    agents = [r['agent'] for r in results_list]

    print(f"\n{'Metric':<40} " + " | ".join(f"{a:>15s}" for a in agents))
    print("-" * 80)

    metrics = [
        ("Total CMDP Cost", "cmdp_cost_total", ".2f"),
        ("- Building SoC Cost", "cmdp_cost_building_soc", ".2f"),
        ("- EV Departure Cost", "cmdp_cost_ev_departure", ".2f"),
        ("", "", ""),
        ("EV Deficit (kWh)", "ev_deficit_total_kwh", ".2f"),
        ("- Avoidable (%)", "ev_deficit_avoidable_pct", ".1f"),
        ("- Unavoidable (kWh)", "ev_deficit_unavoidable_kwh", ".2f"),
        ("", "", ""),
        ("SoC Violations (steps)", "soc_violation_count", "d"),
        ("SoC Violations (%)", "soc_violation_pct", ".1f"),
        ("Battery Abuse (kWh)", "battery_abuse_total_kwh", ".1f"),
        ("Battery Abuse (hours)", "battery_abuse_hours", ".2f"),
        ("", "", ""),
        ("Grid Import (kWh)", "grid_import_total_kwh", ".1f"),
        ("Electricity Cost ($)", "electricity_cost_total", ".2f"),
        ("Carbon Emissions (kg)", "carbon_emissions_kg", ".1f"),
        ("", "", ""),
        ("Total Reward", "total_reward", ".2f"),
    ]

    for metric_name, key, fmt in metrics:
        if not metric_name:
            print("")
            continue

        values = []
        for r in results_list:
            val = r.get(key, 0.0)
            if fmt == "d":
                values.append(f"{int(val):>15d}")
            else:
                values.append(f"{val:>15{fmt}}")

        print(f"{metric_name:<40} " + " | ".join(values))

    print("=" * 80)


def main():
    """Evaluate all baselines and compare."""

    baselines = [
        ("No-Control", "runs/baselines/no_control/kpis.csv"),
        ("Intelligent-RBC", "runs/baselines/intelligent_rbc/kpis.csv"),
        # Add more baselines here as you create them
    ]

    results_list = []
    output_dir = Path("runs/baselines/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # If your environment uses 15-min steps, set step_hours = 0.25, etc.
    STEP_HOURS = 1.0

    for agent_name, kpi_path in baselines:
        if not Path(kpi_path).exists():
            print(f"\n⚠️  Skipping {agent_name}: {kpi_path} not found")
            continue

        results = evaluate_baseline(kpi_path, agent_name, step_hours=STEP_HOURS)
        results_list.append(results)

        with open(output_dir / f"{agent_name.replace(' ', '_').lower()}_eval.json", 'w') as f:
            json.dump(results, f, indent=2)

    if len(results_list) > 1:
        compare_agents(results_list)

    comparison_df = pd.DataFrame(results_list)
    comparison_df.to_csv(output_dir / "baseline_comparison.csv", index=False)

    print(f"\n✅ Evaluation complete!")
    print(f"   Results saved to: {output_dir}")
    print(f"   - Individual JSONs: {agent_name}_eval.json")
    print(f"   - Comparison table: baseline_comparison.csv")


if __name__ == "__main__":
    main()
