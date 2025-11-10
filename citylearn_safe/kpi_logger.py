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
        self.fieldnames = [
            'episode', 'step',
            'obs_mean', 'obs_std', 'obs_min', 'obs_max',
            'soc_mean', 'soc_min', 'soc_max', 'soc_std',
            'action_mean', 'action_std', 'action_min', 'action_max',
            'step_count',
            # Step-wise energy metrics
            'step_net_consumption_kwh', 'grid_import_kwh', 'grid_export_kwh', 'solar_generation_kwh',
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
        self.episode_data = []
        
    def log_step(self, info: Dict[str, Any], step: int, episode: int):
        """Log KPIs from a single step."""
        kpi_data = {
            'episode': episode,
            'step': step,
        }
        
        # Extract KPI keys
        for key, value in info.items():
            if key.startswith(('obs_', 'soc_', 'action_', 'step_', 'total_', 'constraint_', 
                              'grid_', 'solar_', 'electricity_', 'outdoor_', 'indoor_', 'cost', 'reward')):
                kpi_data[key] = float(value) if isinstance(value, (int, float, np.number)) else 0.0
        
        # Also extract cost and reward if available
        if 'cost' in info:
            kpi_data['cost'] = float(info['cost'])
        if 'reward' in info:
            kpi_data['reward'] = float(info['reward'])
        
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
        else:
            # If no episode data, still write any remaining data
            self._write_to_csv()
    
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
    
    def close(self):
        """Close the CSV file."""
        if self.csv_file:
            self._write_to_csv()  # Write any remaining data
            self.csv_file.close()
            self.csv_file = None


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
