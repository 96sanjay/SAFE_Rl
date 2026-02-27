#!/usr/bin/env python3
"""Compare violation rates across agents"""
import pandas as pd

agents = {
    'RBC': 'rbc_baseline.csv',
    'RBC_HeatWave': 'rbc_heatwave.csv',
    # Add your trained agents:
    # 'PPO_Lag': 'ppo_lag_baseline.csv',
    # 'PPO_Lag_HeatWave': 'ppo_lag_heatwave.csv',
}

print("\n" + "="*60)
print("AGENT COMPARISON")
print("="*60)

for name, csv in agents.items():
    try:
        df = pd.read_csv(f"runs/kpi_logs/{csv}")
        viol_rate = df['comfort_violation'].mean() * 100
        print(f"{name:20s}: {viol_rate:5.1f}%")
    except:
        print(f"{name:20s}: (not found)")

print("="*60)
