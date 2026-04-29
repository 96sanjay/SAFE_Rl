MODIFIED SCHEMA: Temperature Control Enabled

Date: December 26, 2024
Base Schema: citylearn_challenge_2022_phase_all_plus_evs
Modified Schema: citylearn_challenge_2022_phase_all_plus_evs_WITH_TEMP_CONTROL

MODIFICATIONS:
- Added cooling_device to all 17 buildings
- Added heating_device to all 17 buildings

DEVICE SPECIFICATIONS:
Cooling Device:
  Type: HeatPump
  Nominal Power: 9.0 kW
  Efficiency (COP): 3.0
  Target Temperature: 24°C

Heating Device:
  Type: HeatPump
  Nominal Power: 9.0 kW
  Efficiency (COP): 3.0
  Target Temperature: 20°C

ACTION SPACE:
Before: 26 actions (17 batteries + 9 EVs)
After: 60 actions (17 batteries + 9 EVs + 17 cooling + 17 heating)

USAGE:
Set environment variable:
  export CITYLEARN_SCHEMA="/home/extra-storage/THESIS/CityLearn/data/datasets/citylearn_challenge_2022_phase_all_plus_evs_WITH_TEMP_CONTROL/schema.json"

Or in Python:
  from citylearn.citylearn import CityLearnEnv
  env = CityLearnEnv(schema="/home/extra-storage/THESIS/CityLearn/data/datasets/citylearn_challenge_2022_phase_all_plus_evs_WITH_TEMP_CONTROL/schema.json")

NOTES:
- This is for testing temperature control feasibility
- All baselines need retraining with new action space
- Temperature constraints can now be actively controlled
