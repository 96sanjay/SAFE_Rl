# R8 STEMS GCN-Transformer Ablation Study

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Determine whether the STEMS GCN-Transformer architecture provides significant improvement over the MLP baseline for 5-building V2G control.

**Architecture:** Pure PyTorch reimplementation of the STEMS spatial-temporal encoder (GCN + Transformer + Gated Fusion) adapted for 5 buildings using ObsIndex-based observation parsing. Integrated into OmniSafe PPOLag via ActorBuilder monkeypatch.

**Tech Stack:** PyTorch (no torch_geometric), OmniSafe PPOLag, CityLearn V2G

---

## Context

R5a (MLP baseline, 50 epochs, 5 buildings) is our best-performing model. We want to test whether replacing the MLP encoder with a GCN-Transformer architecture improves performance. This is a clean ablation: same hyperparameters, same environment, only the policy network architecture changes.

The existing `stems_encoder.py` (689 lines) is designed for 17 buildings and requires `torch_geometric` (not installed). We need a 5-building adaptation with pure PyTorch GCN.

## Architecture

```
obs [B, obs_dim]
  -> ObsIndex parsing -> per-building [B, 5, F_node], global [B, F_global]
  -> ObservationEmbedding (2-layer MLP + LayerNorm) -> [B, 5, 64]
  -> push to HistoryBuffer (24-step window)
  -> AdaptiveGraphConstructor -> learned adj [5, 5]
  -> ResidualGCN (3 layers, pure PyTorch) -> h_spatial [B, 5, 64]
  -> TemporalTransformer (over 24-step window) -> z_temporal [B, 5, 64]
  -> GatedFusion -> r [B, 5, 64]
  -> flatten [B, 320] -> output_proj -> [B, 256]
  -> OmniSafe actor MLP head -> mean actions [B, 9]
```

### Per-Node Features (via ObsIndex)

Each building node receives:
- Building-specific: non_shiftable_load, solar_gen, battery_soc, net_consumption (4 dims)
- EV features (if building has charger): 7 dims, else 7 zeros
- Global broadcast: time(6) + pricing(5) + carbon(1) + washer(2) = 14 dims
- **Per-node raw dim: 4 + 7 + 14 = 25 dims**

Note: Forecast wrapper adds 128 dims that are shared/global. These get broadcast to all nodes as additional global context, increasing per-node dim accordingly.

### Pure PyTorch GCN (replaces torch_geometric)

Standard GCN: `H' = sigma(D_hat^(-1/2) A_hat D_hat^(-1/2) H W)`

```python
class SimpleGCNLayer(nn.Module):
    def forward(self, x, adj_norm):
        # x: [N, D], adj_norm: [N, N] (pre-normalized)
        return self.act(self.linear(adj_norm @ x) + self.skip(x))
```

This is mathematically equivalent to `GCNConv(improved=True)`. No external dependency needed.

## Comparison Design

| Run | Architecture | Config Base | Changes from R5a |
|-----|-------------|-------------|-----------------|
| R5a | MLP [256,256] | r6_compare_r5a_5bld.yaml | None (baseline) |
| R8  | STEMS GCN-Transformer | r8_stems_5bld.yaml | Architecture only |

Both use:
- PPOLag, 50 epochs, seed 42
- Same cost weights, Lagrange params, learning rates
- Same environment (5 buildings, no spatial obs, no C3 controllable)
- obs_normalize: true, reward_normalize: true

## Files

### Create

1. `citylearn_safe/stems_encoder_5bld.py` — Pure PyTorch STEMS encoder for 5 buildings
   - SimpleGCNLayer (replaces torch_geometric GCNConv)
   - ResidualGCNBlock + SpatialEncoder
   - TemporalTransformerBlock (same as original)
   - GatedFusion (same as original)
   - HistoryBuffer (same as original)
   - STEMSEncoder5Bld — main class using ObsIndex for obs parsing
   - OmniSafe actor wrapper (same pattern as train_structured_brain.py)

2. `scripts/train_stems_5bld.py` — Training launcher
   - Builds ObsIndex from env
   - Constructs STEMSEncoder5Bld with correct dimensions
   - Monkeypatches ActorBuilder.build_actor
   - Creates OmniSafe agent and trains

3. `configs/on-policy/r8_stems_5bld.yaml` — Training config (clone of R5a)

4. `run_r8_stems_ablation.sh` — Run script with env vars

### Reference (read-only)

- `citylearn_safe/stems_encoder.py` — Original 17-building STEMS encoder
- `scripts/train_structured_brain.py` — Monkeypatch pattern
- `citylearn_safe/schema_index.py` — ObsIndex
- `configs/on-policy/r6_compare_r5a_5bld.yaml` — R5a config
- `run_r6_comparison_5bld.sh` — R5a env vars

## Expected Outcome

If GCN-Transformer helps: lower cost AND comparable/better reward vs R5a MLP baseline. The spatial message-passing should help coordinate battery actions across buildings.

If it doesn't help: similar or worse metrics, suggesting that for 5 buildings the MLP is sufficient and the GCN-Transformer overhead isn't justified.

---

## Implementation Plan

### Task 1: Create Pure PyTorch STEMS Encoder

**Files:**
- Create: `citylearn_safe/stems_encoder_5bld.py`

**Step 1: Write the encoder module**

Core classes (adapted from `stems_encoder.py`):
- `SimpleGCNLayer` — pure PyTorch GCN (replaces `GCNConv`)
- `ResidualGCNBlock` — GCN + LayerNorm + residual + dropout
- `SpatialEncoder` — stack of ResidualGCN blocks
- `SinusoidalPositionalEncoding` — same as original
- `TemporalTransformerBlock` — same as original
- `GatedFusion` — same as original
- `HistoryBuffer` — same as original
- `ObservationEmbedding` — same as original
- `AdaptiveGraphConstructor5Bld` — simplified (no prior matrices, pure learned)
- `STEMSEncoder5Bld` — main class with ObsIndex-based obs parsing

Key differences from original:
- `SimpleGCNLayer` uses `adj_norm @ x @ W` instead of `GCNConv`
- `_reshape_obs_to_nodes` replaced with `_parse_obs_to_nodes(obs, obs_index)` using exact ObsIndex slicing
- All dimensions parameterized (not hardcoded for 17 buildings)

**Step 2: Verify with quick sanity check**

```bash
cd /home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork
python -c "
from citylearn_safe.stems_encoder_5bld import STEMSEncoder5Bld
import torch
# Quick shape test with dummy obs
enc = STEMSEncoder5Bld(obs_dim=198, num_buildings=5, num_evs=3, hidden_dim=64, output_dim=256)
enc.reset_history(1)
x = torch.randn(1, 198)
y = enc(x)
print(f'Input: {x.shape} -> Output: {y.shape}')
print(f'Params: {sum(p.numel() for p in enc.parameters()):,}')
"
```

**Step 3: Commit**

```bash
git add citylearn_safe/stems_encoder_5bld.py
git commit -m "feat: add pure PyTorch STEMS encoder for 5-building ablation"
```

---

### Task 2: Create Training Script

**Files:**
- Create: `scripts/train_stems_5bld.py`

**Step 1: Write training script**

Pattern from `train_structured_brain.py`:
- Build ObsIndex from local env
- Map building→EV charger associations
- Construct STEMSEncoder5Bld with ObsIndex
- Wrap in CustomGaussianLearningActor
- Monkeypatch ActorBuilder.build_actor
- Parse YAML config, create omnisafe.Agent, train

**Step 2: Smoke test (1 epoch)**

```bash
python scripts/train_stems_5bld.py --cfg configs/on-policy/r8_stems_5bld.yaml --smoke_1epoch
```

Verify:
- Actor class is CustomGaussianLearningActor
- Mean net class is STEMSEncoder5Bld
- Forward pass produces correct action shape [B, 9]
- Training runs for 1 epoch without errors

**Step 3: Commit**

```bash
git add scripts/train_stems_5bld.py
git commit -m "feat: add STEMS 5-building training script with OmniSafe integration"
```

---

### Task 3: Create Config and Run Script

**Files:**
- Create: `configs/on-policy/r8_stems_5bld.yaml`
- Create: `run_r8_stems_ablation.sh`

**Step 1: Create config (clone of R5a)**

```yaml
# R8: STEMS GCN-Transformer ablation on 5 buildings
# Identical to R5a (MLP baseline) except architecture
algo: PPOLag
env_id: CityLearnSafety-V2G-v2
seed: 42

train_cfgs:
  total_steps: 437950          # 50 epochs x 8759 steps/epoch
  vector_env_nums: 1
  parallel: 1

algo_cfgs:
  steps_per_epoch: 8759
  update_iters: 60
  target_kl: 0.10
  kl_early_stop: true
  batch_size: 256
  obs_normalize: true
  reward_normalize: true
  cost_normalize: false
  entropy_coef: 0.005

lagrange_cfgs:
  cost_limit: 24400
  lagrangian_multiplier_init: 10.0
  lambda_lr: 0.05
  lambda_optimizer: SGD

model_cfgs:
  actor:
    hidden_sizes: [256, 256]
    activation: tanh
    lr: 0.0005
  critic:
    hidden_sizes: [256, 256]
    activation: tanh
    lr: 0.001
  linear_lr_decay: false

logger_cfgs:
  use_wandb: false
  use_tensorboard: true
  save_model_freq: 5
  log_dir: ./runs/r8_stems/r8_5bld
  window_lens: 1
```

**Step 2: Create run script**

Same env vars as R5a (from `run_r6_comparison_5bld.sh`) but calls `train_stems_5bld.py`:
- No spatial obs (CITYLEARN_SPATIAL_OBS=0)
- No C3 controllable (CITYLEARN_C3_CONTROLLABLE=0)
- Same cost weights, thresholds, EV settings

**Step 3: Commit**

```bash
git add configs/on-policy/r8_stems_5bld.yaml run_r8_stems_ablation.sh
git commit -m "feat: add R8 STEMS ablation config and run script"
```

---

### Task 4: Run Smoke Test (1 epoch)

**Step 1: Execute smoke test**

```bash
bash run_r8_stems_ablation.sh  # with --smoke_1epoch flag
```

**Step 2: Verify output**

Check:
- Training completes 1 epoch (8759 steps)
- progress.csv has 1 row with valid EpRet, EpCost, EpLen
- StopIter > 1 (no KL early stop bug)
- No NaN/Inf in metrics

**Step 3: Check parameter count**

Compare STEMS encoder params vs MLP baseline to ensure the model isn't dramatically larger.

---

### Task 5: Run Full 50-Epoch Training

**Step 1: Launch training**

```bash
bash run_r8_stems_ablation.sh
```

**Step 2: Monitor early epochs**

Check after 5 epochs:
- StopIter healthy (not stuck at 1)
- EpCost trending down
- No gradient explosions

---

### Task 6: Compare Results

**Step 1: Extract R5a and R8 progress.csv**

**Step 2: Compare key metrics at epoch 49**

| Metric | R5a (MLP) | R8 (STEMS) | Delta |
|--------|-----------|------------|-------|
| Episode Return | | | |
| Episode Cost | | | |
| StopIter | | | |
| Lambda | | | |

**Step 3: Plot learning curves**

Generate comparison plots for reward and cost trajectories.
