import pandas as pd

kpi_file = "runs/kpi_logs/SafePPOLag_Lambda40_HighExplore_100ep.csv"
df = pd.read_csv(kpi_file)

# Get ONLY last episode (last 8759 rows)
df_final = df.tail(8759)

print("="*80)
print("LAMBDA=40 FINAL EPISODE (Epoch 99)")
print("="*80)

total_cost = df_final['cost'].sum()
ev_cost = df_final['cost_ev_departure'].sum()
peak_cost = df_final['cost_grid_peak'].sum()
ramp_cost = df_final['cost_grid_ramp'].sum()

print(f"Total Cost:          {total_cost:.2f}")
print(f"  EV Cost:           {ev_cost:.2f} ({ev_cost/total_cost*100:.1f}%)")
print(f"  Peak Cost:         {peak_cost:.2f} ({peak_cost/total_cost*100:.1f}%)")
print(f"  Ramp Cost:         {ramp_cost:.2f} ({ramp_cost/total_cost*100:.1f}%)")
print()

total_steps = len(df_final)
peak_violations = (df_final['grid_peak_violation'] > 0).sum()
ramp_violations = (df_final['grid_ramp_violation'] > 0).sum()

print(f"Peak Violations:     {peak_violations} ({peak_violations/total_steps*100:.2f}%)")
print(f"Ramp Violations:     {ramp_violations} ({ramp_violations/total_steps*100:.2f}%)")
print()

budget = 350.0
rbc_cost = 304.38
lambda35_cost = 422.9

print(f"Budget:              {budget:.2f}")
print(f"RBC:                 {rbc_cost:.2f}")
print(f"Lambda=35:           {lambda35_cost:.2f}")
print(f"Lambda=40:           {total_cost:.2f}")
print()

if total_cost < budget:
    print("✅ UNDER BUDGET")
elif total_cost < budget * 1.05:
    print("⚠️  Within 5% of budget")
else:
    print(f"❌ Over by {total_cost - budget:.2f} ({(total_cost-budget)/budget*100:.1f}%)")

print("="*80)
