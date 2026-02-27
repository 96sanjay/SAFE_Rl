
"""
UNIFIED Evaluation Pipeline for Safe RL Thesis
Handles BOTH:
  - Detailed KPI CSVs (RBC baselines, evaluated agents)
  - Training progress CSVs (OmniSafe training logs)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
import sys


class EnergySystemEvaluator:
    """Comprehensive evaluation for energy management agents"""
    
    def __init__(self, base_results_dir: str):
        self.base_results_dir = Path(base_results_dir)
        self.results = {}
        self.data_types = {}  # Track which type of data each agent has
        
    def load_agent_results(self, agent_name: str, csv_path: str, data_type: str = 'auto') -> pd.DataFrame:
        """
        Load results for an agent
        
        Args:
            agent_name: Display name
            csv_path: Relative or absolute path to CSV
            data_type: 'kpi' (detailed), 'progress' (training), or 'auto' (detect)
        """
        # Try relative to base_results_dir first
        full_path = self.base_results_dir / csv_path
        if not full_path.exists():
            full_path = Path(csv_path)
        
        if not full_path.exists():
            print(f"❌ ERROR: CSV not found: {csv_path}")
            return None
        
        df = pd.read_csv(full_path)
        
        # Auto-detect data type
        if data_type == 'auto':
            if 'Metrics/EpRet' in df.columns:
                data_type = 'progress'
            elif 'step' in df.columns and 'cost_building_soc' in df.columns:
                data_type = 'kpi'
            else:
                print(f"⚠️  Cannot auto-detect data type for {agent_name}")
                data_type = 'unknown'
        
        self.results[agent_name] = df
        self.data_types[agent_name] = data_type
        
        print(f"✅ Loaded '{agent_name}' ({data_type}): {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def evaluate_all(self, agent_name: str) -> Dict:
        """Compute metrics (handles both data types)"""
        if agent_name not in self.results:
            print(f"❌ Agent '{agent_name}' not loaded!")
            return None
        
        df = self.results[agent_name]
        data_type = self.data_types[agent_name]
        
        if data_type == 'progress':
            return self._evaluate_from_progress(df, agent_name)
        elif data_type == 'kpi':
            return self._evaluate_from_kpi(df, agent_name)
        else:
            print(f"❌ Unknown data type for {agent_name}")
            return None
    
    def _evaluate_from_progress(self, df: pd.DataFrame, agent_name: str) -> Dict:
        """
        Extract metrics from OmniSafe progress.csv
        (Only has summary metrics per epoch, not detailed timestep data)
        """
        # Use the LAST epoch (final trained agent performance)
        last_epoch = df.iloc[-1]
        
        metrics = {
            'agent_name': agent_name,
            'data_type': 'progress',
            'timesteps': int(last_epoch.get('Metrics/EpLen', 0)),
            'epochs_trained': len(df),
            
            # Extract from progress.csv
            'citylearn_kpis': {
                'electricity_consumption': None,  # Not in progress.csv
                'carbon_emissions': None,
                'cost': None,
                'zero_net_energy': None,
                'discomfort_proportion': None,
                'ramping_average': None,
                'daily_peak_average': None,
                'all_time_peak_average': None,
                'citylearn_reward': float(last_epoch.get('Metrics/EpRet', 0)),
            },
            
            'constraint_violations': {
                'total_episode_cost': float(last_epoch.get('Metrics/EpCost', 0)),
                'lagrange_multiplier': float(last_epoch.get('Metrics/LagrangeMultiplier', 0)),
                # Cannot compute detailed violations from progress.csv
                'building_soc': {'violation_percentage': None},
                'ev_departures': {'controllable_percentage': None},
                'grid_peak': {'violation_percentage': None},
                'grid_ramp': {'violation_percentage': None},
                'average_violation_percentage': None,
            },
            
            'energy_metrics': {
                # Not available in progress.csv
                'total_grid_import_kwh': None,
                'total_grid_export_kwh': None,
                'net_consumption_kwh': None,
                'total_solar_generation_kwh': None,
                'solar_waste_kwh': None,
                'solar_utilization_percentage': None,
            },
            
            'cost_breakdown': {
                'total_cmdp_cost': float(last_epoch.get('Metrics/EpCost', 0)),
            },
            
            'per_building': {},  # Not available
            'hourly_analysis': pd.DataFrame(),  # Not available
            
            # Training-specific metrics
            'training_info': {
                'epochs': len(df),
                'initial_cost': float(df.iloc[0].get('Metrics/EpCost', 0)),
                'final_cost': float(last_epoch.get('Metrics/EpCost', 0)),
                'min_cost': float(df['Metrics/EpCost'].min()),
                'mean_cost': float(df['Metrics/EpCost'].mean()),
                'initial_reward': float(df.iloc[0].get('Metrics/EpRet', 0)),
                'final_reward': float(last_epoch.get('Metrics/EpRet', 0)),
                'max_reward': float(df['Metrics/EpRet'].max()),
                'mean_reward': float(df['Metrics/EpRet'].mean()),
            }
        }
        
        return metrics
    
    def _evaluate_from_kpi(self, df: pd.DataFrame, agent_name: str) -> Dict:
        """
        Extract metrics from detailed KPI CSV
        (Has full timestep-by-timestep data)
        """
        metrics = {
            'agent_name': agent_name,
            'data_type': 'kpi',
            'timesteps': len(df),
            
            'citylearn_kpis': self._compute_citylearn_kpis(df),
            'constraint_violations': self._compute_constraint_violations(df),
            'energy_metrics': self._compute_energy_metrics(df),
            'cost_breakdown': self._compute_cost_breakdown(df),
            'per_building': self._compute_per_building_metrics(df),
            'hourly_analysis': self._compute_hourly_analysis(df),
        }
        
        return metrics
    
    # [Keep all the existing _compute_* methods from your original script]
    # I'll include them for completeness:
    
    def _compute_citylearn_kpis(self, df: pd.DataFrame) -> Dict:
        """Extract CityLearn KPIs"""
        kpis = {}
        kpi_cols = {
            'electricity_consumption': 'citylearn_electricity_consumption_total',
            'carbon_emissions': 'citylearn_carbon_emissions_total',
            'cost': 'citylearn_cost_total',
            'zero_net_energy': 'citylearn_zero_net_energy',
            'discomfort_proportion': 'citylearn_discomfort_proportion',
            'ramping_average': 'citylearn_ramping_average',
            'daily_peak_average': 'citylearn_daily_peak_average',
            'all_time_peak_average': 'citylearn_all_time_peak_average',
            'citylearn_reward': 'citylearn_reward',
        }
        
        for key, col in kpi_cols.items():
            if col in df.columns:
                kpis[key] = float(df[col].iloc[-1])
            else:
                kpis[key] = None
        
        return kpis
    
    def _compute_constraint_violations(self, df: pd.DataFrame) -> Dict:
        """Constraint violations (ALL 4)"""
        # SOC
        soc_violations = 0
        if 'cost_building_soc' in df.columns:
            soc_violations = (df['cost_building_soc'] > 0).sum()
        soc_percentage = (soc_violations / len(df)) * 100
        
        # EV
        total_departures = 0
        for i in range(17):
            if f'ev_departures_b{i}' in df.columns:
                total_departures += df[f'ev_departures_b{i}'].sum()
        
        controllable_count = 0
        if 'cost_ev_departure_agent_controllable_v3' in df.columns:
            controllable_count = (df['cost_ev_departure_agent_controllable_v3'] > 0).sum()
        
        ev_controllable_pct = (controllable_count / total_departures * 100) if total_departures > 0 else 0
        
        # Grid Peak
        peak_violations = 0
        peak_percentage = 0.0
        if 'grid_peak_violation' in df.columns:
            peak_violations = int(df['grid_peak_violation'].sum())
            peak_percentage = (peak_violations / len(df)) * 100
        
        # Grid Ramp
        ramp_violations = 0
        ramp_percentage = 0.0
        if 'grid_ramp_violation' in df.columns:
            ramp_violations = int(df['grid_ramp_violation'].sum())
            ramp_percentage = (ramp_violations / len(df)) * 100
        
        avg_violation = (soc_percentage + ev_controllable_pct + peak_percentage + ramp_percentage) / 4.0
        
        return {
            'building_soc': {
                'violation_count': int(soc_violations),
                'violation_percentage': float(soc_percentage),
            },
            'ev_departures': {
                'total_departures': int(total_departures),
                'controllable_count': int(controllable_count),
                'controllable_percentage': float(ev_controllable_pct),
            },
            'grid_peak': {
                'violation_count': peak_violations,
                'violation_percentage': float(peak_percentage),
            },
            'grid_ramp': {
                'violation_count': ramp_violations,
                'violation_percentage': float(ramp_percentage),
            },
            'average_violation_percentage': float(avg_violation),
            'total_episode_cost': float(df['cost'].sum()) if 'cost' in df.columns else 0,
        }
    
    def _compute_energy_metrics(self, df: pd.DataFrame) -> Dict:
        """Energy metrics"""
        total_grid_import = df['grid_import_kwh'].sum() if 'grid_import_kwh' in df.columns else 0
        total_grid_export = df['grid_export_kwh'].sum() if 'grid_export_kwh' in df.columns else 0
        total_solar = df['solar_generation_kwh'].sum() if 'solar_generation_kwh' in df.columns else 0
        net_consumption = df['step_net_consumption_kwh'].sum() if 'step_net_consumption_kwh' in df.columns else 0
        solar_waste = df['solar_waste_kwh'].sum() if 'solar_waste_kwh' in df.columns else 0
        
        solar_used = total_solar - solar_waste if total_solar > 0 else 0
        solar_utilization = (solar_used / total_solar * 100) if total_solar > 0 else 0
        
        return {
            'total_grid_import_kwh': float(total_grid_import),
            'total_grid_export_kwh': float(total_grid_export),
            'net_consumption_kwh': float(net_consumption),
            'total_solar_generation_kwh': float(total_solar),
            'solar_waste_kwh': float(solar_waste),
            'solar_utilization_percentage': float(solar_utilization),
        }
    
    def _compute_cost_breakdown(self, df: pd.DataFrame) -> Dict:
        """Cost breakdown"""
        costs = {}
        if 'cost' in df.columns:
            costs['total_cmdp_cost'] = float(df['cost'].sum())
        if 'cost_building_soc' in df.columns:
            costs['building_soc_cost'] = float(df['cost_building_soc'].sum())
        if 'cost_ev_departure_agent_controllable_v3' in df.columns:
            costs['ev_controllable_cost'] = float(df['cost_ev_departure_agent_controllable_v3'].sum())
        if 'cost_grid_peak' in df.columns:
            costs['grid_peak_cost'] = float(df['cost_grid_peak'].sum())
        if 'cost_grid_ramp' in df.columns:
            costs['grid_ramp_cost'] = float(df['cost_grid_ramp'].sum())
        return costs
    
    def _compute_per_building_metrics(self, df: pd.DataFrame) -> Dict:
        """Per-building breakdown"""
        buildings = {}
        for i in range(17):
            building_metrics = {}
            if f'grid_import_kwh_b{i}' in df.columns:
                building_metrics['grid_import_kwh'] = float(df[f'grid_import_kwh_b{i}'].sum())
            if f'soc_b{i}' in df.columns:
                building_metrics['avg_soc'] = float(df[f'soc_b{i}'].mean())
            if building_metrics:
                buildings[f'building_{i}'] = building_metrics
        return buildings
    
    def _compute_hourly_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hourly analysis"""
        if 'step' not in df.columns:
            return pd.DataFrame()
        
        df_copy = df.copy()
        df_copy['hour'] = df_copy['step'] % 24
        
        hourly = df_copy.groupby('hour').agg({
            'step_net_consumption_kwh': 'mean',
            'grid_import_kwh': 'mean',
        }).round(2)
        
        return hourly
    
    def compare_agents(self, agent_names: List[str]) -> pd.DataFrame:
        """Compare multiple agents"""
        comparison = []
        
        for agent in agent_names:
            if agent not in self.results:
                print(f"⚠️  Skipping '{agent}' (not loaded)")
                continue
            
            metrics = self.evaluate_all(agent)
            if metrics is None:
                continue
            
            # Handle both data types
            data_type = metrics.get('data_type', 'unknown')
            
            if data_type == 'progress':
                row = {
                    'Agent': agent,
                    'Data Type': 'Training Progress',
                    'Total CMDP Cost': metrics['cost_breakdown']['total_cmdp_cost'],
                    'Final Reward': metrics['citylearn_kpis']['citylearn_reward'],
                    'Epochs': metrics.get('epochs_trained', 0),
                    'Lambda': metrics['constraint_violations']['lagrange_multiplier'],
                }
            else:  # kpi
                row = {
                    'Agent': agent,
                    'Data Type': 'Full KPI',
                    'Electricity': metrics['citylearn_kpis'].get('electricity_consumption'),
                    'Carbon': metrics['citylearn_kpis'].get('carbon_emissions'),
                    'Cost': metrics['citylearn_kpis'].get('cost'),
                    'SOC Viol %': metrics['constraint_violations']['building_soc']['violation_percentage'],
                    'EV Viol %': metrics['constraint_violations']['ev_departures']['controllable_percentage'],
                    'Total CMDP Cost': metrics['cost_breakdown'].get('total_cmdp_cost', 0),
                    'Grid Import (kWh)': metrics['energy_metrics']['total_grid_import_kwh'],
                }
            
            comparison.append(row)
        
        return pd.DataFrame(comparison)
    
    def print_summary(self, agent_name: str):
        """Print summary (handles both types)"""
        if agent_name not in self.results:
            print(f"❌ Agent '{agent_name}' not loaded!")
            return
        
        metrics = self.evaluate_all(agent_name)
        if metrics is None:
            return
        
        data_type = metrics.get('data_type', 'unknown')
        
        print(f"\n{'='*90}")
        print(f"  EVALUATION SUMMARY: {agent_name} ({data_type.upper()})")
        print(f"{'='*90}")
        
        if data_type == 'progress':
            # Training progress summary
            print(f"\n📊 Training Info:")
            ti = metrics['training_info']
            print(f"  Epochs trained: {ti['epochs']}")
            print(f"  Final cost:     {ti['final_cost']:.2f} (min: {ti['min_cost']:.2f})")
            print(f"  Final reward:   {ti['final_reward']:.2f} (max: {ti['max_reward']:.2f})")
            print(f"  Lambda:         {metrics['constraint_violations']['lagrange_multiplier']:.2f}")
            
            print(f"\n⚠️  Note: Limited metrics available from progress.csv")
            print(f"     For detailed evaluation, need to run agent and generate KPI CSV")
            
        else:  # kpi
            # Full detailed summary
            kpis = metrics['citylearn_kpis']
            print(f"\n📊 CityLearn KPIs:")
            for key, val in kpis.items():
                if val is not None:
                    print(f"  {key:25s}: {val:.3f}")
            
            cv = metrics['constraint_violations']
            print(f"\n⚠️  Constraint Violations:")
            print(f"  SOC:    {cv['building_soc']['violation_percentage']:5.2f}%")
            print(f"  EV:     {cv['ev_departures']['controllable_percentage']:5.2f}%")
            print(f"  Peak:   {cv['grid_peak']['violation_percentage']:5.2f}%")
            print(f"  Ramp:   {cv['grid_ramp']['violation_percentage']:5.2f}%")
            print(f"  Average:{cv['average_violation_percentage']:5.2f}%")
            
            em = metrics['energy_metrics']
            print(f"\n⚡ Energy:")
            print(f"  Import: {em['total_grid_import_kwh']:,.0f} kWh")
            print(f"  Net:    {em['net_consumption_kwh']:,.0f} kWh")
        
        print(f"{'='*90}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*90)
    print("  UNIFIED SAFE RL EVALUATION PIPELINE")
    print("="*90 + "\n")
    
    evaluator = EnergySystemEvaluator(
        base_results_dir="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/runs"
    )
    
    print("📂 Loading agents...\n")
    
    # RBC Baselines (detailed KPI CSVs)
    evaluator.load_agent_results(
        'Greedy_RBC',
        'baselines/rbc_comparison_COMPLETE/kpis_greedy_COMPLETE.csv',
        data_type='kpi'
    )
    
    evaluator.load_agent_results(
        'Time-Based_RBC',
        'baselines/rbc_comparison_COMPLETE/kpis_time_based_COMPLETE.csv',
        data_type='kpi'
    )
    
    # Trained PPOLag (progress.csv only)
    evaluator.load_agent_results(
        'PPOLag_Trained',
        'ppo_lag_greedy_baseline/PPOLag-{CityLearnSafety-SoC-v0}/seed-000-2026-01-02-18-16-13/progress.csv',
        data_type='progress'
    )
    
    # Print summaries
    print("\n" + "="*90)
    print("  INDIVIDUAL SUMMARIES")
    print("="*90)
    
    for agent_name in evaluator.results.keys():
        evaluator.print_summary(agent_name)
    
    # Comparison table
    print("\n" + "="*120)
    print("  COMPARISON TABLE")
    print("="*120 + "\n")
    
    comparison_df = evaluator.compare_agents(list(evaluator.results.keys()))
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    print(comparison_df.to_string(index=False))
    
    # Save
    output_csv = evaluator.base_results_dir / 'agent_comparison_unified.csv'
    comparison_df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved to: {output_csv}")
    
    print("\n" + "="*90)
    print("  EVALUATION COMPLETE!")
    print("="*90 + "\n")