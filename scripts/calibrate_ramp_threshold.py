#!/usr/bin/env python3
"""
Calibrate ramp threshold from RBC baseline.
Analyzes grid signal change distribution to find safe threshold.
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

def main():
    # Load RBC baseline
    rbc_csv = "runs/kpi_logs/RBC_Greedy_V3.csv"
    
    if not Path(rbc_csv).exists():
        print(f"❌ ERROR: {rbc_csv} not found!")
        print("   Run RBC evaluation first.")
        sys.exit(1)
    
    print("📊 Loading RBC baseline...")
    df = pd.read_csv(rbc_csv)
    
    # Get step_net_consumption_kwh (the signal we ramp on)
    if "step_net_consumption_kwh" not in df.columns:
        print(f"❌ ERROR: 'step_net_consumption_kwh' column not found!")
        sys.exit(1)
    
    net_signal = df["step_net_consumption_kwh"].dropna()
    
    # Compute deltas (absolute change between consecutive timesteps)
    deltas = net_signal.diff().abs().dropna()
    
    print("\n" + "="*60)
    print("GRID RAMP STATISTICS (RBC Baseline)")
    print("="*60)
    print(f"Timesteps analyzed: {len(deltas)}")
    print(f"Mean ramp: {deltas.mean():.2f} kW/hour")
    print(f"Max ramp:  {deltas.max():.2f} kW/hour")
    print(f"Std deviation: {deltas.std():.2f} kW/hour")
    print(f"\nPercentiles:")
    print(f"  90th: {deltas.quantile(0.90):.2f} kW/hour")
    print(f"  95th: {deltas.quantile(0.95):.2f} kW/hour")
    print(f"  97th: {deltas.quantile(0.97):.2f} kW/hour")
    print(f"  99th: {deltas.quantile(0.99):.2f} kW/hour")
    
    # Recommend thresholds
    threshold_conservative = deltas.quantile(0.97)
    threshold_moderate = deltas.quantile(0.95)
    
    print("\n" + "="*60)
    print("RECOMMENDED THRESHOLDS")
    print("="*60)
    
    # Conservative analysis
    violations_conservative = deltas > threshold_conservative
    violations_conservative_count = violations_conservative.sum()
    pct_conservative = violations_conservative_count / len(deltas) * 100
    
    violations_conservative_values = deltas[violations_conservative] - threshold_conservative
    if len(violations_conservative_values) > 0:
        total_cost_conservative = violations_conservative_values.sum()
        mean_violation_conservative = violations_conservative_values.mean()
        max_violation_conservative = violations_conservative_values.max()
    else:
        total_cost_conservative = 0
        mean_violation_conservative = 0
        max_violation_conservative = 0
    
    print(f"\n✅ CONSERVATIVE (97th percentile):")
    print(f"   Threshold: {threshold_conservative:.2f} kW/hour")
    print(f"   Violation count: {violations_conservative_count} steps ({pct_conservative:.1f}%)")
    print(f"   Total cost (if w=1.0): {total_cost_conservative:.2f} kWh")
    print(f"   Mean violation: {mean_violation_conservative:.2f} kW/hour over threshold")
    print(f"   Max violation: {max_violation_conservative:.2f} kW/hour over threshold")
    
    # Moderate analysis
    violations_moderate = deltas > threshold_moderate
    violations_moderate_count = violations_moderate.sum()
    pct_moderate = violations_moderate_count / len(deltas) * 100
    
    violations_moderate_values = deltas[violations_moderate] - threshold_moderate
    if len(violations_moderate_values) > 0:
        total_cost_moderate = violations_moderate_values.sum()
        mean_violation_moderate = violations_moderate_values.mean()
        max_violation_moderate = violations_moderate_values.max()
    else:
        total_cost_moderate = 0
        mean_violation_moderate = 0
        max_violation_moderate = 0
    
    print(f"\n⚡ MODERATE (95th percentile):")
    print(f"   Threshold: {threshold_moderate:.2f} kW/hour")
    print(f"   Violation count: {violations_moderate_count} steps ({pct_moderate:.1f}%)")
    print(f"   Total cost (if w=1.0): {total_cost_moderate:.2f} kWh")
    print(f"   Mean violation: {mean_violation_moderate:.2f} kW/hour over threshold")
    print(f"   Max violation: {max_violation_moderate:.2f} kW/hour over threshold")
    
    print("\n" + "="*60)
    print("RECOMMENDATION FOR IMPLEMENTATION")
    print("="*60)
    print(f"\n👉 Use this in safety_env_v3.py __init__:")
    print(f"   self.ramp_threshold = {threshold_conservative:.2f}")
    print(f"   self.w_grid_ramp = 0.05  # Start small, tune later")
    print(f"\n💡 Expected RBC baseline cost with ramp constraint:")
    print(f"   Conservative: {total_cost_conservative * 0.05:.2f} (with w=0.05)")
    print(f"   Moderate: {total_cost_moderate * 0.05:.2f} (with w=0.05)")
    print("\n   (Start conservative, tighten later if needed)")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
