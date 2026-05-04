# tests/

Integration and smoke tests for the Safe RL V2G pipeline. These verify that
key components initialize correctly, produce expected output shapes, and
preserve backward compatibility. They are not comprehensive unit tests.

## Running

```bash
pytest tests/
```

Or run a single file:

```bash
pytest tests/test_r28_pposaute_config.py -v
```

Some tests are standalone scripts (no pytest fixtures) and can also be run
directly:

```bash
python tests/test_r6_smoke.py
```

## Test Files

| File | Description |
|------|-------------|
| `test_action_projection_serl.py` | SE-RL action projection wrapper: EV-only topology, structural infeasibility reporting, penalty computation inside/outside safe bounds, battery dynamics consistency, QP fastpath discharge power, EV power envelope bracketing, and beta-actor rejection |
| `test_backward_compat.py` | Verifies old behavior is preserved when new feature flags (`CITYLEARN_C3_CONTROLLABLE`, `CITYLEARN_SPATIAL_OBS`) are disabled: C3 costs remain non-negative and observation dimensionality excludes spatial additions |
| `test_controllable_c3.py` | Agent-controllable C3 constraint: a passive (zero-action) agent should incur near-zero controllable C3 cost, and the old (uncontrollable) C3 cost should be strictly greater |
| `test_diagnose_policy_health.py` | Policy Health Diagnostic Suite: synthetic rollout generation, value function critic quality, feature-action mutual information, conditional entropy, action correlation and spatial variance, gradient attribution pathway fractions, temporal planning score, constraint decomposition, headroom vs. baseline, PHI score computation, diagnosis rules, and report generation |
| `test_nsl_nec_indexing.py` | Verifies that non-shiftable load (NSL) and net electricity consumption (NEC) use consistent time indexing in the C3 cost block of `safety_env.py`, ensuring `NEC[t] ~ NSL[t] + solar[t]` for zero-action steps |
| `test_r28_pposaute_config.py` | Validates the R28 PPOSaute YAML config: file existence, required top-level keys, algorithm and env ID, Saute-specific fields and values, normalization settings, epoch/step counts, and full OmniSafe agent instantiation with SauteAdapter observation augmentation |
| `test_r6_smoke.py` | R6 smoke test: runs 500 steps with both spatial observations (P0) and controllable C3 (P1) enabled under random actions, checking for NaN/Inf in observations, finite rewards, and non-zero structural cost removal |
| `test_spatial_obs.py` | Spatial observation wrapper: verifies that enabling `CITYLEARN_SPATIAL_OBS` adds exactly `num_buildings * 4` dimensions to the observation vector and that the appended features are non-zero |

## Notes

- Most tests require the `citylearn` conda environment and the CityLearn
  schema data under `data/citylearn_challenge_2022_phase_all_plus_evs/`.
- Several tests (`test_r6_smoke.py`, `test_backward_compat.py`,
  `test_controllable_c3.py`, `test_nsl_nec_indexing.py`, `test_spatial_obs.py`)
  instantiate the full CityLearn environment, which takes a few seconds per
  test. Use `pytest -x` to stop on first failure during debugging.
- `test_diagnose_policy_health.py` uses synthetic data and does not require the
  CityLearn environment, but does require `scripts/diagnose_policy_health.py`.
- GPU is not required; tests run on CPU by default.
