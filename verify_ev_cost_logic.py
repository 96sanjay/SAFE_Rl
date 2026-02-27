import pandas as pd

print("="*80)
print("EV COST LOGIC VERIFICATION")
print("="*80)

df = pd.read_csv("runs/kpi_logs/Lambda40_FreshEval.csv")

print("\n1. WHAT'S IN THE CSV")
print("-"*80)

# Check all EV deficit columns
print(f"ev_avoidable_deficit_kwh sum:        {df['ev_avoidable_deficit_kwh'].sum():.2f} kWh")
print(f"ev_unavoidable_deficit_kwh sum:      {df['ev_unavoidable_deficit_kwh'].sum():.2f} kWh")
print(f"ev_departure_deficit_kwh sum:        {df['ev_departure_deficit_kwh'].sum():.2f} kWh")

print(f"\ncost_ev_departure_avoidable sum:     {df['cost_ev_departure_avoidable'].sum():.2f}")
print(f"cost_ev_departure_unavoidable sum:   {df['cost_ev_departure_unavoidable'].sum():.2f}")
print(f"cost_ev_departure sum:                {df['cost_ev_departure'].sum():.2f}")

print("\n2. CHECKING IF V3 CONTROLLABLE COLUMNS EXIST")
print("-"*80)

v3_cols = [col for col in df.columns if 'controllable' in col.lower() or 'agent_control' in col.lower()]
if v3_cols:
    print("✅ V3 controllable columns found:")
    for col in v3_cols:
        print(f"  - {col}: sum = {df[col].sum():.2f}")
else:
    print("❌ No V3 controllable columns found")

print("\n3. CHECKING WHAT EV_COST_SCALE SHOULD MULTIPLY")
print("-"*80)

ev_cost_scale = 3.0  # From environment variable
print(f"EV_COST_SCALE = {ev_cost_scale}")

# Test hypothesis: cost_ev_departure = ev_cost_scale × something
cost_ev = df['cost_ev_departure'].sum()
print(f"\nActual cost_ev_departure:     {cost_ev:.2f}")
print(f"If based on avoidable:        {ev_cost_scale * df['ev_avoidable_deficit_kwh'].sum():.2f}")
print(f"If based on total deficit:    {ev_cost_scale * df['ev_departure_deficit_kwh'].sum():.2f}")
print(f"Divided by scale factor:      {cost_ev / ev_cost_scale:.2f}")

print("\n4. TIMESTEP-LEVEL SAMPLE (First 10 steps with EV departures)")
print("-"*80)

sample = df[df['ev_departure_departures'] > 0].head(10)
if len(sample) > 0:
    print(sample[['step', 'ev_departure_departures', 'ev_avoidable_deficit_kwh', 
                  'ev_departure_deficit_kwh', 'cost_ev_departure']].to_string(index=False))
else:
    print("No EV departures found in data")

print("\n5. DIAGNOSIS")
print("-"*80)

if df['cost_ev_departure_avoidable'].sum() == 0 and df['cost_ev_departure_unavoidable'].sum() == 0:
    print("❌ LOGGING BUG: avoidable/unavoidable costs not computed")
    print("   The wrapper computed cost_ev_departure but didn't split it")

if cost_ev > 0 and df['ev_avoidable_deficit_kwh'].sum() > 0:
    print("\n⚠️  MISMATCH DETECTED:")
    print(f"   Avoidable deficit exists: {df['ev_avoidable_deficit_kwh'].sum():.2f} kWh")
    print(f"   But cost_ev_departure_avoidable = 0")
    print(f"   Yet total cost_ev_departure = {cost_ev:.2f}")
    print("\n   This suggests wrapper uses different logic than what's logged")

print("="*80)
