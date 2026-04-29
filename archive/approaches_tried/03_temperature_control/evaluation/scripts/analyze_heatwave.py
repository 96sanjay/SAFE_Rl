import pandas as pd

df = pd.read_csv("runs/kpi_logs/rbc_baseline.csv")

# Assuming outdoor_temperature column exists
if 'outdoor_temperature' not in df.columns:
    print("❌ outdoor_temperature column missing")
    exit(1)

# Filter
normal = df[df['outdoor_temperature'] <= 35]
heatwave = df[df['outdoor_temperature'] > 35]

print("\n" + "="*60)
print("RBC VIOLATION RATES")
print("="*60)
print(f"Normal (≤35°C): {normal['comfort_violation'].mean()*100:.1f}% ({len(normal)} steps)")
print(f"Heat Wave (>35°C): {heatwave['comfort_violation'].mean()*100:.1f}% ({len(heatwave)} steps)")
print(f"Overall: {df['comfort_violation'].mean()*100:.1f}% ({len(df)} steps)")
print("="*60)
