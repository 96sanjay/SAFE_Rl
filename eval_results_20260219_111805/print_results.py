"""Print comparison table from two JSON result files."""
import json
import sys

def load(path):
    with open(path) as f:
        return json.load(f)

def fmt(v, pct=False):
    if isinstance(v, float):
        if pct:
            return f"{v:7.2f}%"
        return f"{v:10.2f}"
    return f"{v:>10}"

def print_table(free_data, psf_data):
    f = free_data["averaged"]
    p = psf_data["averaged"]

    print("\n" + "="*70)
    print(f"{'METRIC':<45} {'FREE RL':>10} {'+ PSF':>10} {'DELTA':>8}")
    print("="*70)

    metrics = [
        ("REWARD", None, None),
        ("Total Reward",           "total_reward",                  False),
        ("Total Bill ($)",         "total_bill_dollars",            False),
        ("Avg Reward/step",        "avg_reward_per_step",           False),
        ("", None, None),
        ("CONSTRAINT VIOLATIONS", None, None),
        ("C1 EV Departure viol%", "c1_ev_departure_violation_rate_%", True),
        ("C1 EV Avoidable kWh",   "c1_ev_avoidable_deficit_kwh",    False),
        ("C2 Battery SoC viol%",  "c2_battery_soc_violation_rate_%", True),
        ("C3 Building Power viol%","c3_building_power_violation_rate_%", True),
        ("C4 Grid Power viol%",   "c4_grid_power_violation_rate_%",  True),
        ("", None, None),
        ("ENERGY", None, None),
        ("Grid Import kWh",        "grid_import_kwh_total",          False),
        ("Grid Export kWh",        "grid_export_kwh_total",          False),
        ("Solar Waste kWh",        "solar_waste_kwh",                False),
        ("", None, None),
        ("PSF STATS", None, None),
        ("PSF Intervention Rate%", "psf_intervention_rate_%",        True),
        ("PSF EV Interventions",   "psf_ev_interventions",           False),
        ("PSF Batt Interventions", "psf_batt_interventions",         False),
        ("PSF Grid Interventions", "psf_grid_interventions",         False),
        ("PSF Infeasible Steps",   "psf_infeasible_steps",           False),
        ("PSF Avg Solve ms",       "psf_avg_solve_ms",               False),
    ]

    for label, key, is_pct in metrics:
        if key is None:
            if label:
                print(f"\n  --- {label} ---")
            else:
                print()
            continue

        fv = f.get(key, float("nan"))
        pv = p.get(key, float("nan"))

        try:
            delta = pv - fv
            delta_str = f"{delta:+8.2f}"
        except Exception:
            delta_str = "       N/A"

        fv_str = f"{fv:10.2f}" + ("%" if is_pct else "")
        pv_str = f"{pv:10.2f}" + ("%" if is_pct else "")
        print(f"  {label:<43} {fv_str:>11} {pv_str:>11} {delta_str:>9}")

    print("="*70)
    print("\nINTERPRETATION:")
    c1_free = f.get("c1_ev_departure_violation_rate_%", 999)
    c1_psf  = p.get("c1_ev_departure_violation_rate_%", 999)
    c2_free = f.get("c2_battery_soc_violation_rate_%", 999)
    c2_psf  = p.get("c2_battery_soc_violation_rate_%", 999)
    c3_free = f.get("c3_building_power_violation_rate_%", 999)
    c3_psf  = p.get("c3_building_power_violation_rate_%", 999)
    c4_free = f.get("c4_grid_power_violation_rate_%", 999)
    c4_psf  = p.get("c4_grid_power_violation_rate_%", 999)

    for name, fv, pv, threshold in [
        ("C1 EV departure", c1_free, c1_psf, 5.0),
        ("C2 Battery SoC",  c2_free, c2_psf, 5.0),
        ("C3 Building pwr", c3_free, c3_psf, 5.0),
        ("C4 Grid power",   c4_free, c4_psf, 5.0),
    ]:
        status_free = "✓" if fv <= threshold else "✗"
        status_psf  = "✓" if pv <= threshold else "✗"
        improved = "↓ improved" if pv < fv else ("= same" if pv == fv else "↑ worse")
        print(f"  {name}: Free={fv:.1f}%{status_free}  PSF={pv:.1f}%{status_psf}  {improved}")

    print()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python print_results.py free_result.json psf_result.json")
        sys.exit(1)
    free_data = load(sys.argv[1])
    psf_data  = load(sys.argv[2])
    print_table(free_data, psf_data)

