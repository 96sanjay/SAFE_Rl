# 07: Pure Lagrangian (No Reward Help)

This approach (R26j) tested whether Lagrangian multipliers alone could enforce
all constraints without reward shaping. It used the same training script and
environment as R25b but with aggressive PID gains and no departure-aware reward.

**Config:** `configs/archive/on-policy/r26j_lagfix.yaml`
**Shell script:** `shell/archive/run_r26j.sh`

No additional source code was needed -- this experiment reused the existing
multi-Lagrangian infrastructure with modified hyperparameters. See the parent
directory's README and `docs/EXPERIMENT_LINEAGE.md` for results and analysis.
