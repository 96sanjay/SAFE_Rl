import pandas as pd

print("="*80)
print("CHECKING ACTUAL COLUMNS IN LOGS")
print("="*80)

# Check Lambda40 columns
df_l40 = pd.read_csv("runs/kpi_logs/Lambda40_FreshEval.csv")
print("\nLambda40 EV-related columns:")
ev_cols_l40 = [col for col in df_l40.columns if 'ev' in col.lower() or 'deficit' in col.lower()]
for col in sorted(ev_cols_l40):
    print(f"  - {col}")

print("\nLambda40 cost columns:")
cost_cols_l40 = [col for col in df_l40.columns if 'cost' in col.lower()]
for col in sorted(cost_cols_l40)[:20]:  # First 20
    print(f"  - {col}")

# Check RBC columns  
df_rbc = pd.read_csv("runs/kpi_logs/RBC_Greedy_V3.csv")
print("\nRBC EV-related columns:")
ev_cols_rbc = [col for col in df_rbc.columns if 'ev' in col.lower() or 'deficit' in col.lower()]
for col in sorted(ev_cols_rbc):
    print(f"  - {col}")

# Now compute correct values
print("\n" + "="*80)
print("CORRECT EV COST BREAKDOWN")
print("="*80)

if 'cost_ev_departure_avoidable' in df_l40.columns:
    avoidable = df_l40['cost_ev_departure_avoidable'].sum()
    unavoidable = df_l40['cost_ev_departure_unavoidable'].sum()
    total_ev = df_l40['cost_ev_departure'].sum()
    
    print(f"\nLambda40:")
    print(f"  Avoidable EV cost:     {avoidable:.2f} kWh (agent's fault)")
    print(f"  Unavoidable EV cost:   {unavoidable:.2f} kWh (physics)")
    print(f"  Total EV cost:         {total_ev:.2f} kWh")
    
    # Check if they match
    if abs(avoidable + unavoidable - total_ev) < 0.01:
        print("  ✅ Avoidable + Unavoidable = Total")
    else:
        print("  ⚠️  Components don't sum to total")

if 'ev_avoidable_deficit_kwh' in df_l40.columns:
    avoidable_deficit = (df_l40['ev_avoidable_deficit_kwh'] > 0).sum()
    print(f"\n  Timesteps with avoidable deficit: {avoidable_deficit}")

if 'cost_ev_departure_avoidable' in df_rbc.columns:
    rbc_avoidable = df_rbc['cost_ev_departure_avoidable'].sum()
    rbc_unavoidable = df_rbc['cost_ev_departure_unavoidable'].sum()
    
    print(f"\nRBC:")
    print(f"  Avoidable EV cost:     {rbc_avoidable:.2f} kWh")
    print(f"  Unavoidable EV cost:   {rbc_unavoidable:.2f} kWh")

print("\n" + "="*80)
print("WHAT THIS MEANS")
print("="*80)
print("Avoidable = Agent could have prevented (TRUE violations)")
print("Unavoidable = Physics/constraints prevented charging")
print("\nIf avoidable cost = 0 → agent charged optimally ✅")
print("If unavoidable > 0 → some EVs departed with unavoidable deficit")
print("="*80)
