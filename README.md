<div align="center">

# SAFE_RL

**Constrained Deep Reinforcement Learning for Vehicle-to-Grid Energy Management**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OmniSafe](https://img.shields.io/badge/OmniSafe-0.4%2B-00B4D8)](https://github.com/PKU-Alignment/omnisafe)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## Abstract

Reinforcement-learning controllers for residential energy districts must respect several safety requirements at once, including electric-vehicle departure readiness, stationary-battery operating limits, per-building power caps, and district-level grid-import ceilings. Standard practice folds all of them into a single shaped reward. Violations then cannot be attributed to a specific requirement or tuned independently.

This repository implements the system described in the thesis *"Development of Constrained Deep Reinforcement Learning Agents for Vehicle-to-Grid Energy Management"*. It routes each safety requirement through its own channel. Every channel carries a dedicated cost signal, cost critic, and enforcement mechanism inside a **four-channel constrained MDP** built on [CityLearn v2](https://github.com/intelligent-environments-lab/CityLearn). Three controllers are introduced: **PPO-Lag-Multi** and **SAC-Lag-Multi** (Multi-Lagrangian methods with independent PID-tuned multipliers per channel) and **CSAC-LB** (a latent log-barrier method). All three are benchmarked against seven OmniSafe baselines and three rule-based controllers over a year-long rollout of a five-building district.

**PPO-Lag-Multi** is the only learning controller that keeps both EV departure violations near zero (0.65%) and battery-safety violations below one per cent. An auxiliary temperature-comfort case study confirms that the pipeline generalises beyond the original four channels.

## Contributions

1. A **four-channel CMDP** for district energy management that retains the operational meaning of each safety requirement throughout training and evaluation.
2. A **Multi-Lagrangian PPO formulation** whose per-channel decoupling is the only mechanism in the thirteen-controller roster that drives EV departure readiness into the safe band while remaining constraint-aware on the per-step channels.
3. Positioning of the **latent log-barrier off-policy controller (CSAC-LB)** as a distinct corner of the Pareto surface -- strong reward performance at the cost of higher constraint violations.
4. An auxiliary **temperature-comfort case study** confirming that the pipeline generalises beyond V2G to additional constraint channels.

## Four Safety Channels

| Channel | Requirement | Type |
|---------|-------------|------|
| C1 | EV departure readiness | Hard deadline, per-departure |
| C2 | Battery safe operation | Bounded state, per-step |
| C3 | Building power envelope | Bounded power, per-step per-building |
| C4 | Grid import ceiling | District cap, per-step |

Each channel has its own cost signal, cost critic (MLP), and PID-controlled Lagrangian multiplier. This per-channel architecture is the central design decision: it prevents inter-constraint interference and makes every violation attributable to a specific operational requirement.

## Pipeline

```
CityLearn Simulation (5 buildings, battery + EV charger)
        |
  CityLearnSafetyEnv  -- cost signals, action mapping, KPI logging
        |
  ForecastObsWrapper   -- 24h price/load/solar lookahead
        |
  TemporalHistoryWrapper -- sliding window of per-building state history
        |
  SauteEVBudgetWrapper -- dense EV budget augmentation (C1)
        |
  CityLearnCMDP       -- OmniSafe CMDP registration, STEMS reward
        |
  PPO-Lag-Multi        -- per-channel PID Lagrangian optimisation
    Actor:  STEMSEncoder -> GaussianPolicy
    Critic: STEMSEncoder (reward) + 4x MLP (per-channel cost)
```

**STEMSEncoder** (GCN + 2-layer Temporal Transformer, 175K parameters) processes each building's state through a per-node temporal transformer (backward history + forward forecast tokens), then aggregates across buildings via an adaptive GCN with gated fusion. The encoder serves the pipeline; the contribution is the per-channel formulation it plugs into, not the encoder architecture itself.

The **headroom-gated reward** (`r_ev_smart`) provides a departure-aware EV price signal: a feasibility corridor opens when the agent has charging slack and closes when departure is urgent, eliminating the reward-constraint conflict that caused prior approaches to fail on C1.

## Repository Structure

```
SAFE_RL/
├── citylearn_safe/                 # Core package
│   ├── cmdp_env.py                 # CMDP environment (OmniSafe-registered)
│   ├── stems_encoder.py            # GCN + Temporal Transformer encoder
│   ├── stems_encoder_5bld.py       # 5-building spatial encoder base
│   ├── pid_lagrange.py             # PID Lagrangian controller
│   ├── safety_env.py               # Safety wrapper (costs, action clipping)
│   ├── extractors.py               # EV departure cost computation, SoC utilities
│   ├── saute_constraint_wrapper.py # Saute EV budget wrapper (C1)
│   ├── grads/                      # Gradient-based algorithms (PPO/SAC Lag-Multi)
│   └── ...
│
├── scripts/
│   ├── training/                   # Training entry points
│   ├── evaluation/                 # Evaluation and metric computation
│   ├── analysis/                   # Post-hoc analysis tools
│   └── utils/                      # Data processing utilities
│
├── configs/
│   ├── active/                     # Current experiment configurations
│   └── archive/                    # Historical configurations
│
├── shell/
│   ├── active/                     # Current launch scripts
│   └── archive/                    # Historical launch scripts
│
├── data/                           # CityLearn datasets
├── evaluation_pipeline/            # Structured evaluation pipeline
├── vendor_deps/                    # Vendored OmniSafe, CVXPyLayers
├── tests/                          # Test suite
├── docs/                           # Extended documentation
└── archive/
    └── approaches_tried/           # Documented explored approaches
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/96sanjay/SAFE_Rl.git
cd SAFE_RL

# Create conda environment
conda create -n citylearn python=3.10 -y
conda activate citylearn

# Install dependencies
pip install torch>=2.0 omnisafe>=0.4 citylearn>=2.1
pip install gymnasium numpy pandas matplotlib tyro pyyaml rich

# Install vendored dependencies
pip install -e vendor_deps/omnisafe
pip install -e vendor_deps/cvxpylayers
```

### Training

```bash
# Using a config file
bash shell/active/run_headroom_gated_cmdp.sh

# Or manually with environment variables
export CITYLEARN_TEMPORAL_WINDOW=12
export CITYLEARN_EV_SAUTE=1
export STEMS_ALPHA_EV_SMART=1.5
python scripts/training/train_multi_lag_stems.py \
    --cfg configs/active/headroom_gated_cmdp.yaml
```

### Evaluation

```bash
# Run thesis evaluation (full benchmark comparison)
python scripts/evaluation/evaluate_thesis.py

# Generate per-departure C1 and per-building C3 violation metrics
python scripts/evaluation/eval_r26hi.py --run-dir runs/headroom_gated_cmdp/
```

## Algorithm Roster

Thirteen controllers benchmarked in the thesis:

| Category | Algorithms |
|----------|-----------|
| Rule-based | Zero-Action, RBC (greedy), RBC (planned) |
| Unconstrained RL | PPO |
| Single-Lagrangian | PPO-Lag, TRPO-Lag, SAC-Lag |
| Projection-based | CPO |
| State-augmented | PPO-Saute |
| **Multi-Lagrangian (ours)** | **PPO-Lag-Multi, SAC-Lag-Multi** |
| **Log-barrier (ours)** | **CSAC-LB** |

PPO-Lag-Multi and SAC-Lag-Multi use independent PID-controlled Lagrangian multipliers (Stooke et al., ICML 2020) per channel. CSAC-LB uses a latent log-barrier formulation for off-policy constraint handling.

## Results

Year-long evaluation on a five-building district (from thesis Table 5.1). C1--C4 columns report constraint violation rates. Deficit is cumulative EV departure energy shortfall.

| Algorithm | EpRet | C1 % | C2 % | C3 % | C4 % | Deficit (kWh) |
|-----------|------:|-----:|-----:|-----:|-----:|--------:|
| Zero-Action | 7230 | 100.00 | 0.00 | 13.25 | 1.77 | 900.17 |
| RBC (greedy) | -3111 | 0.65 | 25.00 | 25.27 | 18.50 | 0.30 |
| PPO | -6432 | 0.65 | 41.05 | 31.92 | 26.64 | 0.30 |
| PPO-Lag | 7825 | 100.00 | 0.00 | 30.47 | 3.62 | 706.26 |
| TRPO-Lag | 6798 | 68.04 | 0.00 | 15.93 | 0.34 | 333.49 |
| CPO | 2503 | 37.94 | 2.44 | 46.49 | 14.33 | 245.15 |
| PPO-Saute | -6559 | 0.65 | 57.50 | 37.02 | 24.71 | 0.30 |
| **PPO-Lag-Multi** | **-4491** | **0.65** | **0.84** | **35.30** | **22.27** | **0.30** |
| SAC-Lag-Multi | -4170 | 11.40 | 14.89 | 55.38 | 29.63 | 27.67 |
| SAC-Lag | -4833 | 13.55 | 26.78 | 53.97 | 28.78 | 17.78 |
| CSAC-LB | 3567 | 69.07 | 3.04 | 19.64 | 4.78 | 361.14 |

**Key finding.** PPO-Lag-Multi is the only learning controller that achieves near-zero C1 violations (0.65%, matching the greedy RBC floor) while simultaneously keeping C2 violations below 1%. Controllers that achieve high EpRet (PPO-Lag, TRPO-Lag, CSAC-LB) do so by neglecting EV departure readiness. CSAC-LB occupies a distinct Pareto position: moderate reward with low C3/C4 violations, but poor C1 compliance.

## Configuration

Training runs are configured through YAML files in `configs/active/`. Key groups:

- **Reward weights** -- Environment variables: `STEMS_MU_ECONOMIC`, `STEMS_BETA_RAMP`, `STEMS_XI_RENEWABLE`, `STEMS_LAMBDA_EV`, `STEMS_ALPHA_EV_SMART`, `STEMS_ALPHA_EV_GUARD`
- **Constraint limits** -- Per-channel cost limits with optional curriculum annealing
- **PID gains** -- Per-channel `Kp`, `Ki`, `Kd` and `penalty_max` (upper bound on multiplier)
- **Encoder** -- Temporal window size, forecast horizon, GCN layers, hidden dimensions
- **Training** -- Epochs, steps per epoch, batch size, learning rate schedules

See [`docs/CONFIGURATION_GUIDE.md`](docs/CONFIGURATION_GUIDE.md) for the full reference.

## Experiment Lineage

The research progressed through systematic experimentation:

| Phase | Runs | Focus |
|-------|------|-------|
| Foundation | R6--R10 | 5-building PPO-Lag, reward tuning, STEMS encoder (R8 ablation, R10 final architecture) |
| Multi-Constraint | R11a--R12a | Single-constraint vs per-constraint Lagrangian, Saute MDP integration |
| EV Safety | R15a--d | EV discharge exploit fix: action clamp, anti-discharge guard, V2G context, PID Ki |
| Curriculum | R18 | Cost-limit annealing (3-phase C1 curriculum) |
| Baseline | R25b | Stable reference configuration (conservative PPO for reporting) |
| Diagnostics | R26f--g | Minimal reward test, STEMS critic validation |
| Reward Shaping | R26h--i | Forecast arbitrage (failed: departure-blind, 70% C1 violation) |
| Lagrangian Stress Test | R26j | Pure Lagrangian without reward help (proved insufficient: lambda saturated) |
| **CMDP Innovation** | **R27a** | **Headroom-gated CMDP with r_ev_smart (`headroom_gated_cmdp.yaml`)** |
| Tuning | R27b--f | r_ev_smart weight and reward combination variants |
| Benchmarks | R28 | OmniSafe stock algorithm comparisons (`run_omnisafe_benchmarks.sh`) |
| Alternative Approaches | R29--R30 | Action masking (`action_mask_simple.yaml`), KL-NEC (`kl_nec_warmstart.yaml`) |

See [`docs/EXPERIMENT_LINEAGE.md`](docs/EXPERIMENT_LINEAGE.md) for per-run configuration diffs.

## Archive: Explored Approaches

The `archive/approaches_tried/` directory documents approaches that were systematically explored during the research. Each informed the final design. The directory preserves original code, configurations, results, and retrospective analysis.

| # | Approach | What it taught |
|---|----------|----------------|
| 01 | Single-constraint Lagrangian | Demonstrated the need for per-channel multipliers when scaling beyond one constraint |
| 02 | Predictive safety filter | Revealed that over-conservative filters collapse exploration; motivated softer budget-based methods |
| 03 | Temperature-comfort extension | Validated that the four-channel pipeline generalises to a fifth constraint type (auxiliary case study in thesis) |
| 04 | Execution shield (hard projection) | Showed that gradient-free action correction prevents the agent from learning constraint structure |
| 05 | SERL action projection | Identified QP infeasibility under V2G dynamics; partially salvaged as a diagnostic tool |
| 06 | Forecast arbitrage reward | Exposed departure-blindness in price-based EV signals; directly motivated `r_ev_smart` |
| 07 | Pure Lagrangian (no reward shaping) | Proved that multiplier-only enforcement saturates before solving hard-deadline constraints like C1 |

## Citation

```bibtex
@mastersthesis{sajeev2026cdrl_v2g,
  title     = {Development of Constrained Deep Reinforcement Learning Agents
               for Vehicle-to-Grid Energy Management},
  author    = {Sajeev, Sanjay},
  school    = {Deggendorf Institute of Technology},
  year      = {2026},
  month     = apr,
  type      = {Master's Thesis ({M.Eng.})},
  note      = {Faculty of Applied Natural Sciences and Industrial Engineering,
               Mechatronic and Cyber-Physical Systems.
               Supervisors: Prof.~Dr.~Andreas Kassler,
               Prof.~Dr.~Christoph Schober}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [CityLearn](https://github.com/intelligent-environments-lab/CityLearn) -- Building energy simulation environment (Vazquez-Canteli et al.)
- [OmniSafe](https://github.com/PKU-Alignment/omnisafe) -- Safe RL framework (PKU-Alignment)
- PID Lagrangian method: Stooke, Achiam, Abbeel. "Responsive Safety in Reinforcement Learning by PID Lagrangian Methods." ICML 2020.
