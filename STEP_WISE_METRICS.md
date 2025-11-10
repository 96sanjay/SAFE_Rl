# Step-Wise Metrics Tracking

## Overview
The KPI logging system now tracks detailed step-by-step metrics in addition to episode-end CityLearn KPIs.

## Step-Wise Metrics (Available Every Timestep)

### Energy Flow
- **`step_net_consumption_kwh`**: Net electricity consumption for this timestep (kWh)
  - Positive = importing from grid
  - Negative = exporting to grid
  
- **`grid_import_kwh`**: Electricity imported from grid (kWh, always ≥ 0)

- **`grid_export_kwh`**: Electricity exported to grid (kWh, always ≥ 0)

- **`solar_generation_kwh`**: Solar PV generation (kWh)

### Economic Metrics
- **`electricity_price`**: Current electricity price ($/kWh)

- **`step_cost`**: Cost for this timestep ($)
  - Calculated as: `step_net_consumption_kwh × electricity_price`

### Environmental
- **`outdoor_temperature`**: Outdoor dry bulb temperature (°C)

- **`indoor_temperature`**: Indoor temperature (°C, if available)

### Safety & Control
- **`soc_mean/min/max/std`**: Battery State of Charge statistics

- **`constraint_violation`**: Binary flag (1.0 if SoC violates bounds, 0.0 otherwise)

- **`cost`**: Safety cost from CMDP formulation

- **`action_mean/std/min/max`**: Control action statistics

## Episode-End CityLearn KPIs (Normalized vs Baseline)

These are only available at episode termination:

- **`citylearn_electricity_consumption_total`**: Total electricity consumption ratio
- **`citylearn_carbon_emissions_total`**: Total carbon emissions ratio  
- **`citylearn_cost_total`**: Total cost ratio
- **`citylearn_zero_net_energy`**: Zero net energy metric
- **`citylearn_daily_peak_average`**: Average daily peak demand
- **`citylearn_discomfort_proportion`**: Proportion of time in discomfort

**Note**: Values > 1.0 indicate worse than baseline, < 1.0 indicate better than baseline

## CSV Output Format

The KPI CSV file (`kpis_kpis.csv`) contains:
1. **Step-by-step rows**: One row per timestep with all step-wise metrics
2. **Episode summary rows**: One row per episode (identified by `step_count=0.0`) with:
   - Averaged step-wise metrics over the episode
   - CityLearn KPIs (only in summary rows)

## Usage Example

```python
import pandas as pd

# Load KPI data
df = pd.read_csv('runs/.../kpis_kpis.csv')

# Get step-by-step data
step_data = df[df['step_count'] != 0.0]

# Get episode summaries
episode_summaries = df[df['step_count'] == 0.0]

# Plot import/export over time
import matplotlib.pyplot as plt
plt.plot(step_data['step_count'], step_data['grid_import_kwh'], label='Import')
plt.plot(step_data['step_count'], step_data['grid_export_kwh'], label='Export')
plt.legend()
plt.show()
```

## Key Insights You Can Now Analyze

1. **Energy Arbitrage**: Track when the agent imports vs exports based on price signals
2. **Temperature Response**: See how outdoor/indoor temperature affects control decisions
3. **Cost Optimization**: Monitor step-wise costs and their relationship to actions
4. **Solar Utilization**: Track how solar generation is used (self-consumption vs export)
5. **Import/Export Patterns**: Identify optimal charging/discharging strategies
6. **Price Response**: Analyze how the agent responds to price fluctuations

## Files Modified

- `citylearn_safe/safety_env.py`: Added step-wise metric computation in `_compute_basic_kpis()`
- `citylearn_safe/kpi_logger.py`: Extended CSV fieldnames to include new metrics
