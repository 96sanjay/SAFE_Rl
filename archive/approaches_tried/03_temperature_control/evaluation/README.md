# Temperature Control Evaluation

## Structure
- `scripts/` - Evaluation scripts
- `results/` - KPI logs (CSV files)

## Usage
```bash
cd scripts
python evaluate_with_kpi.py      # Run agent evaluation
python analyze_heatwave.py        # Analyze heat wave performance
python compare_agents.py          # Compare multiple agents
```

## Results
- RBC Baseline: 41.4% violations (normal), 80.7% (heat wave)
