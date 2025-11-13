"""Custom KPI logger for CityLearn safety environment."""

import csv
import os
from typing import Dict, Any
import numpy as np


class KPILogger:
    """Custom logger to track KPIs separately from OmniSafe."""
    
    def __init__(self, log_dir: str, run_name: str):
        self.log_dir = log_dir
        self.run_name = run_name
        self.csv_path = os.path.join(log_dir, f"{run_name}_kpis.csv")
        self.cost_csv_path = os.path.join(log_dir, f"{run_name}_costs.csv")
        self.fieldnames = [
            'episode', 'step',
            'month', 'day_type', 'non_shiftable_load',
            'soc_mean', 'soc_min', 'soc_max', 'soc_std',
            'action_mean', 'action_std', 'action_min', 'action_max',
            'step_count',
            # Step-wise energy metrics
            'step_net_consumption_kwh', 'grid_import_kwh', 'grid_export_kwh', 'solar_generation_kwh', 'non_shiftable_load_kwh',
            # Step-wise cost and pricing
            'electricity_price', 'step_cost',
            # Temperature tracking
            'outdoor_temperature', 'indoor_temperature',
            # Constraint and cost
            'constraint_violation', 'cost', 'reward',
            # CityLearn KPIs (episode-end only)
            'citylearn_electricity_consumption_total', 'citylearn_carbon_emissions_total',
            'citylearn_cost_total', 'citylearn_daily_peak_average',
            'citylearn_discomfort_proportion', 'citylearn_zero_net_energy'
        ]
        self.csv_file = None
        self.writer = None
        self.cost_csv_file = None
        self.cost_writer = None
        self.episode_summary_csv_path = os.path.join(log_dir, f"{run_name}_episode_summary.csv")
        self.episode_summary_csv_file = None
        self.episode_summary_writer = None
        self.episode_summary_fieldnames = [
            'episode', 'episode_length',
            'total_cost', 'violation_percentage', 'avg_cost_per_step',
            'total_reward', 'avg_reward_per_step',
            'total_import_kwh', 'total_export_kwh', 'total_generation_kwh', 'total_load_kwh',
            'total_electricity_cost', 'carbon_emissions_total',
            'avg_soc', 'min_soc', 'max_soc',
            'avg_electricity_price', 'peak_demand_kw',
            'zero_net_energy', 'discomfort_proportion',
            'lagrangian_multiplier'  # Will be filled from OmniSafe progress.csv if available
        ]
        self.episode_data = []
        self.current_episode_data = []  # Track data for current episode
        
    def log_step(self, info: Dict[str, Any], step: int, episode: int):
        """Log KPIs from a single step."""
        # Reset episode data if this is a new episode (step 0 or 1 after reset)
        if step <= 1 and self.current_episode_data:
            # Check if episode changed
            prev_episode = self.current_episode_data[0].get('episode', 0) if self.current_episode_data else 0
            if episode != prev_episode:
                self.current_episode_data = []
        
        # Track episode data for summary computation
        step_data = {
            'episode': episode,
            'step': step,
            **{k: v for k, v in info.items() if k in self.fieldnames}
        }
        self.current_episode_data.append(step_data)
        
        kpi_data = {
            'episode': episode,
            'step': step,
        }
        
        # Extract KPI keys
        # Explicitly include observation features that don't match prefixes
        explicit_keys = ['month', 'day_type', 'non_shiftable_load']
        for key, value in info.items():
            if key in explicit_keys or key.startswith(('obs_', 'soc_', 'action_', 'step_', 'total_', 'constraint_', 
                              'grid_', 'solar_', 'electricity_', 'outdoor_', 'indoor_', 'cost', 'reward')):
                kpi_data[key] = float(value) if isinstance(value, (int, float, np.number)) else 0.0
        
        # Also extract cost and reward if available
        if 'cost' in info:
            kpi_data['cost'] = float(info['cost'])
        if 'reward' in info:
            kpi_data['reward'] = float(info['reward'])
        
        cost_value = kpi_data.get('cost', None)
        if cost_value is not None:
            print(f"[KPILogger] Episode {episode}, Step {step}: cost={cost_value:.5f}")
            self._log_cost(step=step, episode=episode, cost=cost_value)

        self.episode_data.append(kpi_data)
        
        # Write to CSV every 10 steps for more frequent logging
        if step % 10 == 0:
            self._write_to_csv()
    
    def log_episode_end(self, info: Dict[str, Any], episode: int):
        """Log episode-end KPIs."""
        if self.episode_data:
            # Calculate episode-level statistics using only existing field names
            episode_stats = {
                'episode': episode,
                'step': len(self.episode_data),  # Use step count instead of 'episode_end'
                'obs_mean': 0.0,
                'obs_std': 0.0,
                'obs_min': 0.0,
                'obs_max': 0.0,
                'soc_mean': 0.0,
                'soc_min': 0.0,
                'soc_max': 0.0,
                'soc_std': 0.0,
                'action_mean': 0.0,
                'action_std': 0.0,
                'action_min': 0.0,
                'action_max': 0.0,
                'step_count': 0.0,
                # Step-wise energy metrics
                'step_net_consumption_kwh': 0.0,
                'grid_import_kwh': 0.0,
                'grid_export_kwh': 0.0,
                'solar_generation_kwh': 0.0,
                # Step-wise cost and pricing
                'electricity_price': 0.0,
                'step_cost': 0.0,
                # Temperature
                'outdoor_temperature': 0.0,
                'indoor_temperature': 0.0,
                # Constraint and cost
                'constraint_violation': 0.0,
                'cost': 0.0,
                'reward': 0.0,
                # CityLearn KPIs (episode-end only)
                'citylearn_electricity_consumption_total': 0.0,
                'citylearn_carbon_emissions_total': 0.0,
                'citylearn_cost_total': 0.0,
                'citylearn_daily_peak_average': 0.0,
                'citylearn_discomfort_proportion': 0.0,
                'citylearn_zero_net_energy': 0.0,
            }
            
            # Calculate averages over the episode using existing field names
            # Note: CityLearn KPIs are only available at episode end, so we take the last value
            for key in ['obs_mean', 'obs_std', 'soc_mean', 'soc_min', 'soc_max', 
                       'action_mean', 'action_std', 'constraint_violation', 'cost', 'reward']:
                values = [d.get(key, 0.0) for d in self.episode_data if key in d]
                if values:
                    episode_stats[key] = float(np.mean(values))
            
            # For CityLearn KPIs, use values directly from info dict (only available at episode end)
            for key in ['citylearn_electricity_consumption_total', 'citylearn_carbon_emissions_total',
                       'citylearn_cost_total', 'citylearn_daily_peak_average',
                       'citylearn_discomfort_proportion', 'citylearn_zero_net_energy']:
                if key in info:
                    episode_stats[key] = float(info[key])
            
            # Use constraint_violation field for episode summary
            violations = sum(1 for d in self.episode_data if d.get('constraint_violation', 0) > 0)
            episode_stats['constraint_violation'] = violations / len(self.episode_data) if self.episode_data else 0.0
            
            self.episode_data.append(episode_stats)
            self._write_to_csv()
            
            # Clear episode data
            self.episode_data = []
            
            # Compute and log episode summary
            self._log_episode_summary(episode, info)
        else:
            # If no episode data, still write any remaining data
            self._write_to_csv()
    
    def _log_episode_summary(self, episode: int, info: Dict[str, Any]):
        """Compute and log episode-level summary statistics."""
        if not self.current_episode_data:
            return
        
        # Filter to step rows (exclude episode summaries)
        step_data = [d for d in self.current_episode_data if d.get('step', 0) > 0]
        if not step_data:
            self.current_episode_data = []
            return
        
        summary = {
            'episode': episode,
            'episode_length': len(step_data),
        }
        
        # Total cost and violation percentage
        costs = [d.get('cost', 0.0) for d in step_data]
        violations = [d.get('constraint_violation', 0.0) for d in step_data]
        summary['total_cost'] = float(sum(costs))
        summary['violation_percentage'] = float(sum(violations) / len(step_data) * 100.0) if step_data else 0.0
        summary['avg_cost_per_step'] = float(np.mean(costs)) if costs else 0.0
        
        # Total reward
        rewards = [d.get('reward', 0.0) for d in step_data]
        summary['total_reward'] = float(sum(rewards))
        summary['avg_reward_per_step'] = float(np.mean(rewards)) if rewards else 0.0
        
        # Energy metrics
        imports = [d.get('grid_import_kwh', 0.0) for d in step_data]
        exports = [d.get('grid_export_kwh', 0.0) for d in step_data]
        generation = [d.get('solar_generation_kwh', 0.0) for d in step_data]
        
        summary['total_import_kwh'] = float(sum(imports))
        summary['total_export_kwh'] = float(sum(exports))
        summary['total_generation_kwh'] = float(sum([abs(g) for g in generation]))  # Solar is negative
        # Total load from CityLearn (electricity consumption total)
        summary['total_load_kwh'] = float(info.get('citylearn_electricity_consumption_total', 0.0))
        
        # Electricity cost and carbon
        summary['total_electricity_cost'] = float(info.get('citylearn_cost_total', 0.0))
        summary['carbon_emissions_total'] = float(info.get('citylearn_carbon_emissions_total', 0.0))
        
        # SoC statistics
        soc_means = [d.get('soc_mean', 0.0) for d in step_data if 'soc_mean' in d]
        soc_mins = [d.get('soc_min', 0.0) for d in step_data if 'soc_min' in d]
        soc_maxs = [d.get('soc_max', 0.0) for d in step_data if 'soc_max' in d]
        summary['avg_soc'] = float(np.mean(soc_means)) if soc_means else 0.0
        summary['min_soc'] = float(np.min(soc_mins)) if soc_mins else 0.0
        summary['max_soc'] = float(np.max(soc_maxs)) if soc_maxs else 0.0
        
        # Average electricity price
        prices = [d.get('electricity_price', 0.0) for d in step_data if 'electricity_price' in d]
        summary['avg_electricity_price'] = float(np.mean(prices)) if prices else 0.0
        
        # Peak demand (from CityLearn or computed)
        summary['peak_demand_kw'] = float(info.get('citylearn_daily_peak_average', 0.0))
        
        # CityLearn metrics
        summary['zero_net_energy'] = float(info.get('citylearn_zero_net_energy', 0.0))
        summary['discomfort_proportion'] = float(info.get('citylearn_discomfort_proportion', 0.0))
        
        # Lagrangian multiplier (will be filled from progress.csv later if needed)
        summary['lagrangian_multiplier'] = 0.0
        
        # Write to episode summary CSV
        if self.episode_summary_csv_file is None:
            self.episode_summary_csv_file = open(self.episode_summary_csv_path, 'w', newline='')
            self.episode_summary_writer = csv.DictWriter(
                self.episode_summary_csv_file,
                fieldnames=self.episode_summary_fieldnames
            )
            self.episode_summary_writer.writeheader()
        
        filtered_summary = {k: v for k, v in summary.items() if k in self.episode_summary_fieldnames}
        self.episode_summary_writer.writerow(filtered_summary)
        self.episode_summary_csv_file.flush()
        
        # Clear current episode data
        self.current_episode_data = []
    
    def _write_to_csv(self):
        """Write accumulated data to CSV file."""
        if not self.episode_data:
            return
            
        # Initialize CSV file if needed
        if self.csv_file is None:
            # Use the predefined fieldnames instead of inferring from data
            self.csv_file = open(self.csv_path, 'w', newline='')
            self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
            self.writer.writeheader()
        
        # Write data (filter to only include predefined fieldnames)
        for row in self.episode_data:
            filtered_row = {k: v for k, v in row.items() if k in self.fieldnames}
            self.writer.writerow(filtered_row)
        
        self.csv_file.flush()
        self.episode_data = []
    
    def _log_cost(self, *, step: int, episode: int, cost: float) -> None:
        """Write per-step cost to a dedicated CSV."""
        if self.cost_csv_file is None:
            self.cost_csv_file = open(self.cost_csv_path, 'w', newline='')
            self.cost_writer = csv.DictWriter(
                self.cost_csv_file,
                fieldnames=['episode', 'step', 'cost'],
            )
            self.cost_writer.writeheader()
        self.cost_writer.writerow({'episode': episode, 'step': step, 'cost': cost})
        self.cost_csv_file.flush()
    
    def close(self):
        """Close the CSV file."""
        if self.csv_file:
            self._write_to_csv()  # Write any remaining data
            self.csv_file.close()
            self.csv_file = None
        if self.cost_csv_file:
            self.cost_csv_file.close()
            self.cost_csv_file = None
        if self.episode_summary_csv_file:
            self.episode_summary_csv_file.close()
            self.episode_summary_csv_file = None


# Global logger instance
_kpi_logger = None

def init_kpi_logger(log_dir: str, run_name: str):
    """Initialize the global KPI logger."""
    global _kpi_logger
    _kpi_logger = KPILogger(log_dir, run_name)

def log_kpis(info: Dict[str, Any], step: int, episode: int):
    """Log KPIs using the global logger."""
    global _kpi_logger
    if _kpi_logger:
        _kpi_logger.log_step(info, step, episode)

def log_episode_end(info: Dict[str, Any], episode: int):
    """Log episode-end KPIs."""
    global _kpi_logger
    if _kpi_logger:
        _kpi_logger.log_episode_end(info, episode)

def close_kpi_logger():
    """Close the global KPI logger."""
    global _kpi_logger
    if _kpi_logger:
        _kpi_logger.close()
        _kpi_logger = None
