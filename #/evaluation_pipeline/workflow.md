# Unified Evaluation Workflow

## Step 1: Generate Baseline (RBC)
```bash
python3 evaluate_rbc_with_v3_CORRECT.py
# Output: rbc_greedy_v3_results.csv
# Output: runs/kpi_logs/RBC_Greedy_V3.csv
```

## Step 2: Evaluate Trained Agents
For each trained model, generate detailed KPI CSV:
```bash
python3 evaluate_trained_agent.py \
  --checkpoint runs/ppo_lag_bill_reward_35ep/PPOLag*/seed*/torch_save/epoch-35.pt \
  --output runs/evaluated_agents/bill_reward_e35_kpis.csv
```

## Step 3: Unified Comparison
```bash
python3 unified_evaluator.py \
  --agents \
    "RBC_Greedy:runs/kpi_logs/RBC_Greedy_V3.csv:kpi" \
    "PPOLag_CostOnly:runs/evaluated_agents/cost_only_e85_kpis.csv:kpi" \
    "PPOLag_BillBased:runs/evaluated_agents/bill_reward_e35_kpis.csv:kpi" \
  --output comparison_table.csv
```

## Step 4: Case Study Analysis
Organize results by case study:

### CS1: Sanity Check
- Verify V3 missing_actions = 0 for all agents
- Check reward/cost signals make sense

### CS2: Main Comparison (Reward Functions)
- RBC Greedy
- PPOLag Cost-Only (85 epochs)
- PPOLag Bill-Based (35 epochs)
- PPOLag Multi-Objective (planned)

Compare: cost, reward, battery usage, exports, violations

### CS3: Ablation Study (Constraints)
- Config 1: Only SOC constraint
- Config 2: SOC + EV
- Config 3: SOC + EV + Peak
- Config 4: SOC + EV + Peak + Ramp

Show how each constraint affects performance

### CS4: Stress Test
- Tighter budgets (e.g., 50% of baseline)
- Different scenarios (high demand days)
- Sensitivity analysis
