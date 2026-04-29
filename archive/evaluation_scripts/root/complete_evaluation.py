import pandas as pd

print("="*80)
print("COMPLETE EVALUATION: LAMBDA=40 vs RBC BASELINE")
print("="*80)

# Load Lambda=40 final episode
kpi_file = "runs/kpi_logs/SafePPOLag_Lambda40_HighExplore_100ep.csv"
df_lambda40 = pd.read_csv(kpi_file).tail(8759)

# Load RBC baseline
rbc_file = "runs/kpi_logs/RBC_Greedy_V3.csv"
df_rbc = pd.read_csv(rbc_file)

print("\n1. TOTAL COSTS")
print("-"*80)

lambda40_total = df_lambda40['cost'].sum()
lambda40_ev = df_lambda40['cost_ev_departure'].sum()
lambda40_peak = df_lambda40['cost_grid_peak'].sum()
lambda40_ramp = df_lambda40['cost_grid_ramp'].sum()

rbc_total = df_rbc['cost'].sum()
rbc_ev = df_rbc['cost_ev_departure'].sum()
rbc_peak = df_rbc['cost_grid_peak'].sum()
rbc_ramp = df_rbc['cost_grid_ramp'].sum()

print(f"{'Metric':<20} {'RBC':>12} {'Lambda=40':>12} {'Delta':>12}")
print(f"{'Total Cost':<20} {rbc_total:>12.2f} {lambda40_total:>12.2f} {lambda40_total-rbc_total:>+12.2f}")
print(f"{'EV Cost':<20} {rbc_ev:>12.2f} {lambda40_ev:>12.2f} {lambda40_ev-rbc_ev:>+12.2f}")
print(f"{'Peak Cost':<20} {rbc_peak:>12.2f} {lambda40_peak:>12.2f} {lambda40_peak-rbc_peak:>+12.2f}")
print(f"{'Ramp Cost':<20} {rbc_ramp:>12.2f} {lambda40_ramp:>12.2f} {lambda40_ramp-rbc_ramp:>+12.2f}")

print("\n2. VIOLATION COUNTS")
print("-"*80)

lambda40_peak_viol = (df_lambda40['grid_peak_violation'] > 0).sum()
lambda40_ramp_viol = (df_lambda40['grid_ramp_violation'] > 0).sum()
lambda40_ev_viol = (df_lambda40['ev_controllable_deficit_kwh'] > 0).sum() if 'ev_controllable_deficit_kwh' in df_lambda40.columns else 0

rbc_peak_viol = (df_rbc['grid_peak_violation'] > 0).sum()
rbc_ramp_viol = (df_rbc['grid_ramp_violation'] > 0).sum()
rbc_ev_viol = 0

print(f"{'Constraint':<20} {'RBC':>12} {'Lambda=40':>12} {'Delta':>12}")
print(f"{'EV Violations':<20} {rbc_ev_viol:>12} {lambda40_ev_viol:>12} {lambda40_ev_viol-rbc_ev_viol:>+12}")
print(f"{'Peak Violations':<20} {rbc_peak_viol:>12} {lambda40_peak_viol:>12} {lambda40_peak_viol-rbc_peak_viol:>+12}")
print(f"{'Ramp Violations':<20} {rbc_ramp_viol:>12} {lambda40_ramp_viol:>12} {lambda40_ramp_viol-rbc_ramp_viol:>+12}")

print("\n3. VIOLATION PERCENTAGES")
print("-"*80)

total_steps = 8760
lambda40_peak_pct = lambda40_peak_viol / total_steps * 100
lambda40_ramp_pct = lambda40_ramp_viol / total_steps * 100
lambda40_ev_pct = lambda40_ev_viol / total_steps * 100

rbc_peak_pct = rbc_peak_viol / total_steps * 100
rbc_ramp_pct = rbc_ramp_viol / total_steps * 100
rbc_ev_pct = 0.0

print(f"{'Constraint':<20} {'RBC':>12} {'Lambda=40':>12} {'Delta':>12}")
print(f"{'EV Violations':<20} {rbc_ev_pct:>11.2f}% {lambda40_ev_pct:>11.2f}% {lambda40_ev_pct-rbc_ev_pct:>+11.2f}%")
print(f"{'Peak Violations':<20} {rbc_peak_pct:>11.2f}% {lambda40_peak_pct:>11.2f}% {lambda40_peak_pct-rbc_peak_pct:>+11.2f}%")
print(f"{'Ramp Violations':<20} {rbc_ramp_pct:>11.2f}% {lambda40_ramp_pct:>11.2f}% {lambda40_ramp_pct-rbc_ramp_pct:>+11.2f}%")

print("\n4. VIOLATION MAGNITUDES (Mean)")
print("-"*80)

lambda40_peak_mag = df_lambda40[df_lambda40['cost_grid_peak_raw'] > 0]['cost_grid_peak_raw'].mean() if lambda40_peak_viol > 0 else 0
lambda40_ramp_mag = df_lambda40[df_lambda40['cost_grid_ramp_raw'] > 0]['cost_grid_ramp_raw'].mean() if lambda40_ramp_viol > 0 else 0

rbc_peak_mag = df_rbc[df_rbc['cost_grid_peak_raw'] > 0]['cost_grid_peak_raw'].mean() if rbc_peak_viol > 0 else 0
rbc_ramp_mag = df_rbc[df_rbc['cost_grid_ramp_raw'] > 0]['cost_grid_ramp_raw'].mean() if rbc_ramp_viol > 0 else 0

print(f"{'Constraint':<20} {'RBC':>12} {'Lambda=40':>12}")
print(f"{'Peak (kW over)':<20} {rbc_peak_mag:>12.2f} {lambda40_peak_mag:>12.2f}")
print(f"{'Ramp (kW/h over)':<20} {rbc_ramp_mag:>12.2f} {lambda40_ramp_mag:>12.2f}")

print("\n5. BUDGET ASSESSMENT")
print("-"*80)

budget = 350.0
print(f"Cost Budget:         {budget:.2f}")
print(f"RBC Cost:            {rbc_total:.2f} ({(rbc_total-budget)/budget*100:+.1f}%)")
print(f"Lambda=40 Cost:      {lambda40_total:.2f} ({(lambda40_total-budget)/budget*100:+.1f}%)")

print("\n6. SUMMARY")
print("-"*80)
if lambda40_total < budget:
    print("✅ Lambda=40: UNDER BUDGET")
elif lambda40_total < budget * 1.05:
    print("⚠️  Lambda=40: Within 5% of budget")
else:
    print(f"❌ Lambda=40: Over budget by {lambda40_total - budget:.2f}")

if lambda40_total < rbc_total:
    print(f"✅ Lambda=40 beats RBC by {rbc_total - lambda40_total:.2f}")
else:
    print(f"❌ RBC beats Lambda=40 by {lambda40_total - rbc_total:.2f}")

print("="*80)
