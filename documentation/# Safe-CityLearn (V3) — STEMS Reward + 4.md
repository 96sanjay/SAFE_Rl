# Safe-CityLearn (V3) — STEMS Reward + 4-Constraint CMDP Cost
**Last updated:** 2026-01-21  
**Project:** Safe Reinforcement Learning for V2G Energy Management (CityLearn 2022 + EVs)  
**Student:** Sanjay Sajeev (Copycat)  
**Repo root:** `/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/`

This document is a **single source of truth** for what has been implemented so far, including:
- STEMS-style reward (without comfort term)
- Safety constraints (4 constraints total)
- How constraints are computed, logged, calibrated, and combined into CMDP cost
- KPI logging schema and verification checks
- Environment variables controlling behavior

---

## 0) Files & Where Things Live

### Core wrapper (environment)
- `citylearn_safe/safety_env_v3.py`
  - Class: `CityLearnSafetyEnvV3`
  - Implements:
    - Action clipping (per-dimension)
    - EV deficit classification (V3 action-based)
    - STEMS reward (economic + stability + renewable)
    - 4 safety constraints (EV + battery SOC + building power + grid import)
    - CMDP cost composition for OmniSafe (PPO-Lagrangian)

### KPI logging
- `citylearn_safe/kpi_logger.py`
  - Class: `KPILogger`
  - Writes:
    - `runs/kpi_logs/<RUN_NAME>.csv` (per step KPIs, 100+ columns)
    - `runs/kpi_logs/<RUN_NAME>_costs.csv` (per step cost breakdown)
    - `runs/kpi_logs/<RUN_NAME>_episode_summary.csv` (episode summary)

### Dataset
- `data/citylearn_challenge_2022_phase_all_plus_evs/schema.json` (set via `CITYLEARN_SCHEMA`)

---

## 1) High-level Design

### 1.1 Reward (training objective)
Two reward modes exist:
- **Bill reward** (baseline CityLearn-style): import cost minus export revenue
- **STEMS reward** (implemented): multi-objective **economic + stability + renewable**
  - **Comfort term is NOT included** (no HVAC/temperature control actions in this dataset/setup).

Switch via:
- `CITYLEARN_REWARD_TYPE="bill"` (default)
- `CITYLEARN_REWARD_TYPE="stems"`

### 1.2 Safety (constraints for Safe RL / CMDP cost)
Final goal: **4 constraints** in CMDP cost (for PPO-Lagrangian):
1. **EV departure shortfall (agent-controllable, V3)**
2. **Battery SOC safety band** (STEMS Eq. 16 style, implemented as hinge outside [SOC_LOW, SOC_HIGH])
3. **Building power cap** (STEMS Eq. 17 style, per-building |power| cap)
4. **Grid import cap** (STEMS Eq. 18 style, district import cap)

Important:
- These constraints are computed per-step and aggregated into a **single scalar** `info["cost"]` for OmniSafe.
- For per-building constraints (battery SOC, building power), aggregation uses **soft-max via p-norm** (Option C).

---

## 2) Environment Variables (Control Knobs)

### 2.1 Required
```bash
export CITYLEARN_SCHEMA="/.../data/.../schema.json"
export PYTHONPATH="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork:$PYTHONPATH"
```

### 2.2 Reward selection & economics
```bash
export CITYLEARN_REWARD_TYPE="stems"   # or "bill"
export CITYLEARN_EXPORT_FACTOR="0.7"  # export paid at export_factor × import price
export CITYLEARN_REWARD_SCALE="1.0"
```

### 2.3 EV deficit / cost scaling
```bash
export CITYLEARN_EV_COST_SCALE="3.0"             # scales EV controllable deficit cost
export CITYLEARN_EV_MISSING_ACTION_MODE="assume_full"  # assume_full | assume_zero | error
export CITYLEARN_INCLUDE_EV_COST="1"             # include EV in CMDP cost (wrapper default: True)
```

### 2.4 STEMS constraint thresholds
Battery SOC band:
```bash
export CITYLEARN_STEMS_SOC_LOW="0.05"
export CITYLEARN_STEMS_SOC_HIGH="0.95"
```

p-norm softness (shared by battery + building constraints):
```bash
export CITYLEARN_STEMS_PNORM_P="4.0"   # p=4 default; larger -> closer to max
```

Building power cap (per building kW):
```bash
export CITYLEARN_STEMS_P_BUILDING_MAX="4.6083"   # calibrated from no-control (pooled p97)
```

Grid import cap (district kW):
```bash
export CITYLEARN_STEMS_P_GRID_MAX="29.6915"      # calibrated from no-control (p97)
```

### 2.5 CMDP cost weights (to balance magnitudes)
These weights multiply raw constraint magnitudes when forming `info["cost"]`:
```bash
export CITYLEARN_W_COST_EV="1.0"
export CITYLEARN_W_COST_SOC="10.0"
export CITYLEARN_W_COST_BUILDING="0.5"
export CITYLEARN_W_COST_GRID="0.05"
```

### 2.6 KPI logging
```bash
export CITYLEARN_KPI_RUN_NAME="MY_RUN"
export CITYLEARN_KPI_FLUSH_EVERY_STEP="1"    # 1 = flush every step (debug); else flush every 100 steps
```

---

## 3) STEMS Reward Implementation (No Comfort)

### 3.1 Reward structure
STEMS reward implemented as:
- `reward_economic`
- `reward_stability` (grid + building + ramp)
- `reward_renewable`
- `reward_stems_total = reward_economic + reward_stability + reward_renewable`

### 3.2 Economic term (correct export factor)
Economic component uses **import and export explicitly**:
\[
R_{econ} = -\mu \cdot price \cdot (import - export\_factor \cdot export)
\]
Implementation reads:
- `export_factor = float(os.environ.get("CITYLEARN_EXPORT_FACTOR","1.0"))`

This fixes the earlier bug where export was treated as equally valuable as import.

### 3.3 Stability term (indirectly controllable)
- Depends on grid import ratio, average per-building consumption ratio, and ramp penalty.
- Even without a “stability action,” battery/EV actions affect `net_consumption` and `grid_import`.

Ramp for STEMS reward uses internal memory:
- `self._prev_net_consumption`
- reset sets `self._prev_net_consumption = None`

### 3.4 Renewable term (district solar sum)
Solar generation is summed **across all buildings** (not only building_0), and `abs()` is used because solar may be stored with negative sign convention.

---

## 4) Safety Constraints (4 total)

### 4.1 Constraint 1 — EV departure shortfall (V3, agent-controllable)
Computed using:
- `ev_departure_cost_components_v3(...)` from `citylearn_safe/extractors_v3.py`
- Uses stored action history aligned to tau = (pre-step time_step + 1)
- Splits:
  - `agent_controllable` (used for training signal)
  - `uncontrollable` (physics residual)

Raw magnitudes logged:
- `ev_departure_deficit_kwh` (total deficit)
- `ev_avoidable_deficit_kwh` (controllable)
- `ev_unavoidable_deficit_kwh` (uncontrollable)

CMDP EV cost uses:
- `ev_cost_for_cmdp = ev_cost_scale * ev_agent_control_v3` (if `include_ev_in_cost`)

### 4.2 Constraint 2 — Battery SOC safety (STEMS Eq. 16 style)
Per building:
\[
v_i = \max(0, SOC_{low} - SOC_i) + \max(0, SOC_i - SOC_{high})
\]

Aggregation (soft-max p-norm):
\[
c_{soc} = \left(\frac{1}{N}\sum_i v_i^p\right)^{1/p}
\]
with `p = CITYLEARN_STEMS_PNORM_P` (default 4).

Logged fields:
- `cost_stems_battery` (p-norm aggregated magnitude)
- `battery_soc_violation` (1 if ANY building violates; strict alarm)

Note:
- `battery_soc_violation` is strict (district OR). This is useful for monitoring, but can remain 1 until **all** buildings are within band.

### 4.3 Constraint 3 — Building power cap (STEMS Eq. 17 style)
Per building:
- `p_i = building.net_electricity_consumption[idx]` (kW)
- `v_i = max(0, |p_i| - P_building_max)`

Aggregation (soft-max p-norm):
\[
c_{bld} = \left(\frac{1}{N}\sum_i v_i^p\right)^{1/p}
\]

Logged fields:
- `cost_stems_building_power` (p-norm aggregated)
- `building_power_violation` (1 if ANY building violates)

Threshold:
- `P_building_max = CITYLEARN_STEMS_P_BUILDING_MAX` (kW)

### 4.4 Constraint 4 — Grid import cap (STEMS Eq. 18 style)
District-level scalar (no p-norm needed):
\[
v_{grid} = max(0, grid\_import - P_{grid,max})
\]

Logged fields:
- `cost_stems_grid_power`
- `grid_power_violation`

Threshold:
- `P_grid_max = CITYLEARN_STEMS_P_GRID_MAX` (kW)

---

## 5) CMDP Cost (What OmniSafe uses)

### 5.1 Raw components (unweighted)
- EV: `cost_ev_departure`  (already includes `CITYLEARN_EV_COST_SCALE`)
- Battery: `cost_stems_battery`
- Building power: `cost_stems_building_power`
- Grid: `cost_stems_grid_power`

### 5.2 Weighted CMDP cost
In `step()`:
```python
total_cost = w_ev*ev_cost_for_cmdp + w_soc*c_soc + w_bld*c_bld + w_grid*c_grid
info["cost"] = total_cost
```

Weights via env vars:
- `CITYLEARN_W_COST_EV`
- `CITYLEARN_W_COST_SOC`
- `CITYLEARN_W_COST_BUILDING`
- `CITYLEARN_W_COST_GRID`

### 5.3 Legacy constraints (still logged, not used in CMDP cost)
Your older grid operational costs remain computed/logged (if present in code):
- `cost_grid_peak`, `cost_grid_ramp` + flags
but they are **not part of** `info["cost"]` after the 4-constraint CMDP change (unless changed later).

---

## 6) KPI Logger: Schema & Guarantees

### 6.1 Key design guarantees (KPILogger)
- Fieldnames are defined once in `__init__` and treated as stable schema
- Auto-upgrade: if existing CSV header differs, file is rewritten preserving old rows
- Optional flush every step: `CITYLEARN_KPI_FLUSH_EVERY_STEP=1`
- Action logging always includes:
  - `action_0..action_25`
  - `action_ev_0..action_ev_7`
- Reward fields always include both bill + STEMS components (even if reward_type is bill)

### 6.2 Added/required constraint columns (4 constraints)
In main CSV header:
- `cost_ev_departure`
- `cost_stems_battery`, `battery_soc_violation`
- `cost_stems_building_power`, `building_power_violation`
- `cost_stems_grid_power`, `grid_power_violation`
- plus total CMDP `cost`

---

## 7) Calibration Results (Policy-free: No-control baseline)

### 7.1 Building power cap calibration (no-control, full year)
From 8760 steps, pooled building-time pairs:
- pooled |p_i| p97 = **4.6083 kW**  (~3% building-time violations)

Also (per-step max across buildings):
- max_t |p| p97 = 9.1033 kW (useful if you want a “timestep max” style threshold)

Selected for per-building cap:
- `CITYLEARN_STEMS_P_BUILDING_MAX=4.6083`

### 7.2 Grid import cap calibration (no-control, full year)
From 8759 steps:
- grid_import p97 = **29.6915 kW** (~3% timestep violations)

Selected:
- `CITYLEARN_STEMS_P_GRID_MAX=29.6915`

---

## 8) Verification & Debug Runs Performed

### 8.1 One-step numeric checks
- Verified economic reward uses `CITYLEARN_EXPORT_FACTOR`
- Verified ramp term in STEMS reward is non-zero after step 1
- Verified grid constraint triggers at step 1 with calibrated grid cap:
  - step 1 grid_import 51.574 => grid_cost 21.8829 => flag 1

### 8.2 200-step CSV integrity check
A 200-step run confirmed:
- CSV updated with rows and columns
- Non-zero occurrences:
  - `cost` nonzero for all steps
  - `cost_stems_battery` nonzero for all steps (no-control)
  - building/grid violations appear occasionally
- Preview rows show correct values and reward_type “stems”

### 8.3 Full-year baseline cost (no-control)
Example full-year totals with weights (w_soc=10, w_bld=0.5, w_grid=0.05, w_ev=1):
- Steps: 8759
- Total weighted episode cost: **22682.997**
- Unweighted component sums:
  - EV: 16844.455
  - SOC: 437.95
  - BLD: 2801.0005
  - GRID: 1170.8389
- Violation steps:
  - SOC: 8759
  - BLD: 2549
  - GRID: 263

Interpretation:
- No-control makes EV very bad (agent never charges EVs), so EV dominates. This baseline is for scaling only, not “good performance.”

---

## 9) Notes / Caveats

### 9.1 SOC values at episode start can be zeros
Observed: at reset, battery SOC time series values at index 0 are often 0.0 for buildings.
This causes SOC constraint violation early unless actions charge batteries.

This is expected given the dataset initialization and is not an extraction bug.

### 9.2 “ANY building violated” flags are strict
- `battery_soc_violation` and `building_power_violation` are strict OR across buildings.
- With many buildings, these flags can remain 1 even if most buildings are safe.

For analysis, consider also logging future metrics (optional):
- violation rate = (# buildings violating)/N
- max violation magnitude across buildings
These are not required for training but help interpret results.

### 9.3 Cost_limit must be recalibrated after changing weights/cost definition
PPO-Lagrangian `cost_limit` must match the new scale of `info["cost"]`.
Recommended workflow:
1) Choose a baseline policy you trust (your planned baseline script).
2) Run a full episode, get baseline episode cost.
3) Set `cost_limit = baseline_cost * (1.10 to 1.15)`.

---

## 10) Quick “Known Good” Env Var Bundle (Current)
```bash
export CITYLEARN_REWARD_TYPE="stems"
export CITYLEARN_EXPORT_FACTOR="0.9"
export CITYLEARN_REWARD_SCALE="1.0"

export CITYLEARN_STEMS_SOC_LOW="0.05"
export CITYLEARN_STEMS_SOC_HIGH="0.95"
export CITYLEARN_STEMS_PNORM_P="4.0"
export CITYLEARN_STEMS_P_BUILDING_MAX="4.6083"
export CITYLEARN_STEMS_P_GRID_MAX="29.6915"

export CITYLEARN_W_COST_EV="1.0"
export CITYLEARN_W_COST_SOC="10.0"
export CITYLEARN_W_COST_BUILDING="0.5"
export CITYLEARN_W_COST_GRID="0.05"

export CITYLEARN_KPI_FLUSH_EVERY_STEP="0"
export CITYLEARN_KPI_RUN_NAME="MY_EXPERIMENT"
```

---

## 11) Change Log (What was fixed/added)

### Reward fixes
- Renewable solar sum across all buildings (abs sign fix)
- STEMS ramp term fixed using `self._prev_net_consumption`
- Economic term fixed to respect `CITYLEARN_EXPORT_FACTOR`
- `self._prev_net_consumption` reset in `reset()`

### Constraints added
- Battery SOC band constraint (p-norm soft-max)
- Building power cap constraint (p-norm soft-max)
- Grid import cap constraint (hinge)
- All 3 logged to KPI CSV with flags

### CMDP cost changes
- `info["cost"]` now includes **4 constraints** with weights:
  EV + SOC + Building + Grid
- Legacy peak/ramp costs remain logged but are not included in CMDP cost (unless changed later)

### KPI logger updates
- Added new columns:
  - `cost_stems_battery`, `battery_soc_violation`
  - `cost_stems_building_power`, `building_power_violation`
  - `cost_stems_grid_power`, `grid_power_violation`
- Logging verified with 200-step CSV test

---

## 12) Next Steps (Planned)
1) Run your baseline policy evaluation for 8760 steps and compute baseline episode cost with the new cost.
2) Choose `cost_limit` based on that baseline (e.g., +10% slack).
3) Train PPOLag with the new `info["cost"]` definition.
4) Evaluate on multiple seeds and report:
   - episode cost, reward, violation counts, distributions
   - compare against baseline policy