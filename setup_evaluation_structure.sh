#!/bin/bash

BASE_DIR="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$BASE_DIR" || { echo "Directory not found"; exit 1; }

echo "Setting up professional evaluation structure..."

# 1. Main evaluation pipeline
mkdir -p evaluation_pipeline/{scripts,configs,raw_results,processed_tables,plots,reports}

# 2. Case studies structure
mkdir -p evaluation_pipeline/case_studies/{cs1_sanity,cs2_main_comparison,cs3_ablation,cs4_stress_test}

# 3. Baseline controllers
mkdir -p evaluation_pipeline/baselines/{rbc_greedy,rbc_time_based,no_control}

# 4. Trained agents results
mkdir -p evaluation_pipeline/agents/{cost_only,bill_based,multi_objective}

# 5. Analysis outputs
mkdir -p evaluation_pipeline/analysis/{per_building,hourly,constraint_breakdown,v3_verification}

# 6. Comparison outputs
mkdir -p evaluation_pipeline/comparisons/{rbc_vs_agents,reward_comparison,constraint_ablation}

# 7. Raw data from runs (symlink to avoid duplication)
ln -sf "$BASE_DIR/runs/kpi_logs" evaluation_pipeline/raw_results/kpi_logs
ln -sf "$BASE_DIR/runs" evaluation_pipeline/raw_results/training_runs

echo "✅ Structure created!"
tree -L 3 evaluation_pipeline/

