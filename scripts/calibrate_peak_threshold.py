#!/usr/bin/env python3
"""
Calibrate peak threshold from RBC baseline.
Analyzes grid import distribution to find safe threshold.
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
    
    # Get grid import column
    if "grid_import_kwh" not in df.columns:
        print(f"❌ ERROR: 'grid_import_kwh' column not found!")
        print(f"   Available columns: {list(df.columns[:10])}")
        sys.exit(1)
    
    imports = df["grid_import_kwh"].dropna()
    
    print("\n" + "="*60)
    print("GRID IMPORT STATISTICS (RBC Baseline)")
    print("="*60)
    print(f"Timesteps analyzed: {len(imports)}")
    print(f"Mean import:        {imports.mean():.2f} kW")
    print(f"Max import:         {imports.max():.2f} kW")
    print(f"Std deviation:      {imports.std():.2f} kW")
    print(f"\nPercentiles:")
    print(f"  90th: {imports.quantile(0.90):.2f} kW")
    print(f"  95th: {imports.quantile(0.95):.2f} kW")
    print(f"  97th: {imports.quantile(0.97):.2f} kW")
    print(f"  99th: {imports.quantile(0.99):.2f} kW")
    
    # Recommend thresholds
    threshold_conservative = imports.quantile(0.97)
    threshold_moderate = imports.quantile(0.95)
    
    print("\n" + "="*60)
    print("RECOMMENDED THRESHOLDS")
    print("="*60)
    
    # Conservative analysis
    violations_conservative_mask = imports > threshold_conservative
    violations_conservative_count = violations_conservative_mask.sum()
    pct_conservative = violations_conservative_count / len(imports) * 100
    
    if violations_conservative_count > 0:
        violations_conservative_values = imports[violations_conservative_mask] - threshold_conservative
        total_cost_conservative = violations_conservative_values.sum()
        mean_violation_conservative = violations_conservative_values.mean()
        max_violation_conservative = violations_conservative_values.max()
    else:
        total_cost_conservative = 0
        mean_violation_conservative = 0
        max_violation_conservative = 0
    
    print(f"\n✅ CONSERVATIVE (97th percentile):")
    print(f"   Threshold: {threshold_conservative:.2f} kW")
    print(f"   Violation count: {violations_conservative_count} steps ({pct_conservative:.1f}%)")
    print(f"   Total cost (if w=1.0): {total_cost_conservative:.2f} kWh")
    print(f"   Mean violation: {mean_violation_conservative:.2f} kW over threshold")
    print(f"   Max violation: {max_violation_conservative:.2f} kW over threshold")
    
    # Moderate analysis
    violations_moderate_mask = imports > threshold_moderate
    violations_moderate_count = violations_moderate_mask.sum()
    pct_moderate = violations_moderate_count / len(imports) * 100
    
    if violations_moderate_count > 0:
        violations_moderate_values = imports[violations_moderate_mask] - threshold_moderate
        total_cost_moderate = violations_moderate_values.sum()
        mean_violation_moderate = violations_moderate_values.mean()
        max_violation_moderate = violations_moderate_values.max()
    else:
        total_cost_moderate = 0
        mean_violation_moderate = 0
        max_violation_moderate = 0
    
    print(f"\n⚡ MODERATE (95th percentile):")
    print(f"   Threshold: {threshold_moderate:.2f} kW")
    print(f"   Violation count: {violations_moderate_count} steps ({pct_moderate:.1f}%)")
    print(f"   Total cost (if w=1.0): {total_cost_moderate:.2f} kWh")
    print(f"   Mean violation: {mean_violation_moderate:.2f} kW over threshold")
    print(f"   Max violation: {max_violation_moderate:.2f} kW over threshold")
    
    print("\n" + "="*60)
    print("RECOMMENDATION FOR STEP 4")
    print("="*60)
    print(f"\n👉 Use this in safety_env_v3.py __init__:")
    print(f"   self.peak_threshold = {threshold_conservative:.2f}")
    print(f"   self.w_grid_peak = 0.05  # Start small, tune later")
    print(f"\n💡 Expected RBC baseline cost with peak constraint:")
    print(f"   Conservative: {total_cost_conservative * 0.05:.2f} (with w=0.05)")
    print(f"   Moderate: {total_cost_moderate * 0.05:.2f} (with w=0.05)")
    print("\n   (Start conservative, tighten later if needed)")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
