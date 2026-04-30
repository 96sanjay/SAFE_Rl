# 03: Temperature Control Case Study

> **Status**: Completed thesis chapter (not a failed approach)
>
> This is a self-contained case study on safe RL for HVAC temperature control.
> It was completed and written up as a thesis chapter, but the V2G (Vehicle-to-Grid)
> focus became the primary contribution. All code, figures, and LaTeX sources
> are preserved here for reproducibility.

## Overview

**Problem**: Control HVAC systems in 5 buildings to minimize energy consumption
while maintaining thermal comfort (indoor temperature within acceptable bounds).

**Approach**: Constrained RL with multiple algorithms — PPO-Lag, SAC-Lag, CPO,
FOCOPS, CUP, CSAC-LB — benchmarked with multi-seed evaluation.

**Key finding**: CSAC-LB (Conservative SAC with Lyapunov Barrier) achieved the
best safety-efficiency tradeoff for temperature control, with near-zero
constraint violations after 40 epochs.

## Directory Structure

```
03_temperature_control/
├── README.md                          # This file
│
├── # ── Environment & Algorithm Code ──
├── omni_env_temp.py                   # Base temperature control environment
├── omni_env_temp_cooling_only.py      # Cooling-only variant (main)
├── omni_env_temp_masked.py            # Action-masked variant
├── omni_env_temp_masked_reward.py     # Masked reward variant
├── ppo_lag_temp_cooling_only.py       # PPO-Lag adaptation
├── ppo_lag_temp_masked.py             # PPO-Lag masked variant
├── ppo_temp_masked.py                 # PPO masked variant
├── sac_lag_temp_cooling_only.py       # SAC-Lag adaptation
├── policy_action_mask_temp.py         # Temperature-specific action mask
│
├── # ── Training Scripts ──
├── scripts/
│   ├── train_ppolag_temp_cooling_only.py
│   ├── train_ppolag_temp_masked.py
│   ├── train_ppolag_temp.py
│   ├── train_ppo_temp_masked.py
│   ├── train_saclag_temp_cooling_only.py
│   ├── train_benchmark_temp_cooling_only.py  # Multi-algorithm benchmark
│   └── train_csac_lb_temp.py                 # CSAC-LB training
│
├── # ── Evaluation Scripts ──
├── eval_scripts/
│   ├── evaluate_csaclb_temp_case_study.py    # CSAC-LB evaluation
│   ├── evaluate_ppo_temp_case_study.py       # PPO evaluation
│   └── select_temp_feasible_checkpoint.py    # Checkpoint selection
│
├── # ── Shell Launch Scripts ──
├── shell/
│   ├── run_ppolag_temp_cooling_only.sh
│   ├── run_ppolag_temp_cooling_only_tight_v2.sh
│   ├── run_ppolag_temp_cooling_only_tight_v3.sh
│   ├── run_ppolag_temp_cooling_only_neverblock_v4.sh
│   ├── run_saclag_temp_cooling_only.sh
│   └── run_benchmark_temp_cooling_only.sh
│
├── # ── YAML Configs (25 variants) ──
├── configs/
│   ├── csac_lb_temp_benchmark.yaml           # Main benchmark config
│   ├── ppolag_temp_cooling_only_*.yaml       # PPO-Lag variants
│   ├── csac_lb_temp_*.yaml                   # CSAC-LB variants
│   └── ...
│
├── # ── Thesis Figures (PDFs) ──
├── figures/
│   ├── latex_ready/                   # Final figures used in thesis (48 PDFs)
│   │   ├── fig_benchmark_training_curves.pdf
│   │   ├── fig_benchmark_violation_rates.pdf
│   │   ├── fig_benchmark_pareto_front.pdf
│   │   ├── fig_csaclb_paper_pareto_scatter_combined.pdf
│   │   ├── fig_multiseed_robustness.pdf
│   │   ├── fig_stress_test_violation.pdf
│   │   └── ...
│   ├── paper_style/                   # Paper-formatted figures (11 PDFs)
│   ├── ablation_eval/                 # Ablation study figures + data
│   │   ├── fig1_constraint_violations.pdf
│   │   ├── rollout_grads.npz          # Raw rollout data
│   │   └── ...
│   └── constraint_conflict/           # Constraint conflict analysis
│       ├── fig1_cost_correlation.pdf
│       ├── rollout_data.npz
│       └── ...
│
├── # ── Figure Generation Scripts ──
├── figure_scripts/                    # Reproduce ALL thesis figures
│   ├── plot_algo_taxonomy.py
│   ├── plot_benchmark_all_algos.py
│   ├── plot_csaclb_paper_style.py
│   ├── plot_energy_vs_temp_deviation.py
│   ├── plot_multiseed_summary.py
│   ├── plot_stress_test.py
│   ├── regen_fig_*.py                 # Regeneration scripts
│   └── compute_significance.py        # Statistical significance tests
│
├── # ── Analysis ──
├── analysis/
│   ├── r28a_vs_r25b/                  # R28a vs baseline comparison
│   │   ├── compare_r25b_r28a.py
│   │   └── fig1-7_*.pdf
│   └── r28b_reward_analysis/          # Reward decomposition
│       ├── analyze_reward_terms.py
│       └── fig1-7_*.pdf
│
├── # ── LaTeX Source ──
├── latex/
│   ├── thesis_temperature_case_study.tex      # Chapter source
│   ├── thesis_temperature_case_study_v2.tex   # Revised version
│   ├── main.tex                               # Standalone compilation
│   └── references.bib                         # Bibliography
│
├── # ── Reports ──
├── ablation_eval_report.md
├── constraint_conflict_report.md
├── temperature_case_study_benchmark_report.md
├── THESIS_TEMPERATURE_CASE_STUDY_REVIEW.md
├── MASTER_THESIS_STRUCTURE.md
│
├── # ── Data ──
├── data/
│   ├── all_checkpoint_evals.json      # All checkpoint evaluation results
│   └── tb_ablation_data.npz           # TensorBoard ablation data
│
└── evaluation/                        # Evaluation pipeline results
```

## How to Reproduce

### 1. Training (example: CSAC-LB benchmark)
```bash
# From project root
source shell/archive/run_benchmark_temp_cooling_only.sh
```

### 2. Evaluation
```bash
python archive/approaches_tried/03_temperature_control/eval_scripts/evaluate_csaclb_temp_case_study.py
```

### 3. Regenerate Figures
```bash
# All figure generation scripts are in figure_scripts/
python archive/approaches_tried/03_temperature_control/figure_scripts/plot_benchmark_all_algos.py
```

### 4. Dataset
The temperature control dataset is at:
```
data/citylearn_challenge_2022_phase_all_plus_evs_WITH_TEMP_CONTROL/
```

## Algorithms Benchmarked

| Algorithm | Type | Key Result |
|-----------|------|------------|
| PPO-Lag | On-policy, Lagrangian | Good reward, moderate violations |
| SAC-Lag | Off-policy, Lagrangian | Fast convergence, some violations |
| CPO | On-policy, trust region | Conservative, low violations |
| FOCOPS | On-policy, first-order | Balanced performance |
| CUP | On-policy, constrained | Moderate performance |
| **CSAC-LB** | **Off-policy, Lyapunov** | **Best safety-efficiency tradeoff** |

## Lessons Learned

1. **Off-policy methods converge faster** for temperature control (SAC-based > PPO-based)
2. **Lyapunov barrier functions** (CSAC-LB) provide stronger safety guarantees than Lagrangian
3. **Multi-seed evaluation is essential** — single-seed results were misleading for CPO
4. **Domain-specific reward shaping** matters — generic energy minimization conflicts with comfort
5. **Cooling-only formulation** is more tractable than full HVAC control

## Relationship to Main V2G Work

The temperature case study was completed first and informed several design decisions
for the V2G work:
- PID Lagrangian (developed here) was carried forward to V2G
- Multi-constraint framework was validated here before applying to 5-constraint V2G
- Benchmark methodology (multi-seed, stress test) was established here
