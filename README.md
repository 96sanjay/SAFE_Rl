# Safe-CityLearn: Safe RL Benchmarking for Building Energy Management

Benchmarking suite for Safe Reinforcement Learning algorithms on multi-building energy optimization with battery safety constraints.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo>
cd safe-citylearn

# Install dependencies
pip install -r requirements.txt

# Install OmniSafe (separately)
pip install omnisafe

# Install CityLearn
pip install CityLearn
```

### 2. Environment Setup

```bash
# Set schema path
export CITYLEARN_SCHEMA="$PWD/data/citylearn/schema.json"
```

### 3. Test Installation

```bash
# Quick environment test
python -m scripts.quick_env_smoke

# Test safety wrapper
python -m scripts.safety_smoke
```

### 4. Train a Single Algorithm

```bash
# Train PPOLag (recommended starting point)
python -m scripts.train_omnisafe --cfg configs/on-policy/ppo_lag_soc.yaml
```

### 5. Run Full Benchmark

```bash
# Train all 11 algorithms sequentially
python -m scripts.benchmark_algos
```

## Algorithms Supported

### On-Policy Methods
| Algorithm | Type | Config |
|-----------|------|--------|
| PPO | Baseline (no safety) | `ppo_soc.yaml` |
| PPOLag | Lagrangian | `ppo_lag_soc.yaml` |
| TRPO | Trust Region | `trpo_soc.yaml` |
| TRPOLag | Lagrangian | `TRPO_lag_soc.yaml` |
| CPO | Constrained | `cpo_soc.yaml` |
| PCPO | Projection | `pcpo_soc.yaml` |
| RCPO | Reward-Constrained | `RCPO_soc.yaml` |

### Off-Policy Methods
| Algorithm | Type | Config |
|-----------|------|--------|
| SAC | Baseline | `sac.yaml` |
| SACLag | Lagrangian | `saclag.yaml` |
| TD3 | Twin Delayed | `td3.yaml` |
| TD3Lag | Lagrangian | `td3lag.yaml` |
| DDPG | Deterministic | `ddpg.yaml` |
| DDPGLag | Lagrangian | `ddpglag.yaml` |

## Results

View with TensorBoard:
```bash
tensorboard --logdir runs/
```

Key Metrics:
- `Metrics/EpRet`: Episode return (energy cost)
- `Metrics/EpCost`: Episode constraint violation
- `Metrics/LagrangeMultiplier`: Constraint enforcement strength (Lagrangian methods)
- `Train/KL`: Policy update magnitude

## Customization

### 1. Select Specific Algorithms

Edit `scripts/benchmark_algos.py`:

```python
ALGOS = [
    "PPOLag",    # Lagrangian baseline
    "CPO",       # Constrained optimization
    "SACLag",    # Off-policy Lagrangian
]
```

### 2. Adjust Training Budget

```python
BASE = {
    "env_id": "CityLearnSafety-SoC-v0",
    "train_cfgs": {
        "total_steps": 87590,      # 10 epochs → Change to 175180 for 20 epochs
        "vector_env_nums": 1
    },
    "algo_cfgs": {
        "steps_per_epoch": 8759,   # 1 episode = 1 epoch (don't change)
        "obs_normalize": True,
        "reward_normalize": True
    }
}
```

### 3. Modify Constraint Limit

Create custom config:

```yaml
# configs/on-policy/ppo_lag_relaxed.yaml
algo: PPOLag
env_id: CityLearnSafety-SoC-v0

train_cfgs:
  total_steps: 87590

algo_cfgs:
  steps_per_epoch: 8759
  obs_normalize: true
  reward_normalize: true

lagrange_cfgs:
  cost_limit: 0.10          # ← Relaxed (was 0.05)
  lagrangian_multiplier_init: 1.0  # ← Stronger initial penalty
```

## Citation

```bibtex
@misc{safe-citylearn,
  title={Safe Reinforcement Learning for Building Energy Management},
  author={Your Name},
  year={2025},
  url={https://github.com/...}
}
```

## License

MIT License

