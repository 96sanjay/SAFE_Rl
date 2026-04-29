#!/usr/bin/env python3
"""Extract default CityLearn KPIs from episode summary"""
import pandas as pd

summary = pd.read_csv("../results/rbc_baseline_episode_summary.csv")

print("\n" + "="*70)
print("RBC - DEFAULT CITYLEARN KPIs")
print("="*70)

print(f"\n[ENERGY]")
print(f"  Total consumption: {summary['total_load_kwh'].mean():.1f} kWh")
print(f"  Total import: {summary['total_import_kwh'].mean():.1f} kWh")
print(f"  Total export: {summary['total_export_kwh'].mean():.1f} kWh")
print(f"  Total generation: {summary['total_generation_kwh'].mean():.1f} kWh")

print(f"\n[COST & CARBON]")
print(f"  Total cost: ${summary['total_electricity_cost'].mean():.2f}")
print(f"  Carbon emissions: {summary['carbon_emissions_total'].mean():.1f} kg CO2")

print(f"\n[GRID METRICS]")
print(f"  Peak demand: {summary['peak_demand_kw'].mean():.2f} kW")
print(f"  Ramping score: {summary['ramping_score'].mean():.4f}")
print(f"  Zero net energy: {summary['zero_net_energy'].mean():.4f}")

print(f"\n[COMFORT]")
print(f"  Discomfort proportion: {summary['discomfort_proportion'].mean():.4f}")

print(f"\n[EPISODE STATS]")
print(f"  Avg episode length: {summary['episode_length'].mean():.0f} steps")
print(f"  Avg reward: {summary['avg_reward_per_step'].mean():.4f}")

print("\n" + "="*70 + "\n")
