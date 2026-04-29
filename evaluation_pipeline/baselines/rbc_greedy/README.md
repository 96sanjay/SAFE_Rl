# RBC Greedy Baseline

## Configuration
- **Battery Control:** Time-of-use
  - Charge: 10:00-16:00 (solar hours)
  - Discharge: 17:00-21:00 (peak hours)
- **EV Control:** Greedy charging (action = 1.0 always)
- **Washing Machine:** Default behavior

## Key Results
- Total Cost: 0.00 (zero violations!)
- EV Controllable Deficit (V3): 0.00 kWh
- EV Uncontrollable Deficit (V3): 71.66 kWh
- V3 Missing Actions: 0 ✅

## Files
- `summary.csv` - One-row summary of key metrics
- `full_kpis.csv` - Full timestep data (8760 steps × 98 columns)
- `README.md` - This file

## Generated
Date: 2026-01-08
Script: `evaluation_pipeline/scripts/evaluate_rbc_with_v3_CORRECT.py`
