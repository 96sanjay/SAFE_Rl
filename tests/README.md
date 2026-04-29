# tests/

Integration and smoke tests for the Safe RL V2G pipeline. These verify that
key components initialize correctly and produce expected output shapes. They
are not comprehensive unit tests.

## Running

```bash
pytest tests/
```

Or run a single file:

```bash
pytest tests/test_omni_env.py -v
```

## Test Files

| File | Description |
|------|-------------|
| `test_omni_env.py` | Smoke test for CityLearnCMDP registration, reset, and step |
| `test_safety_env.py` | Verifies CityLearnSafetyEnv cost signals (C0-C4) |
| `test_stems_encoder.py` | Forward pass through STEMSEncoder with dummy obs |
| `test_pid_lagrange.py` | PID controller update logic, anti-windup, and curriculum annealing |
| `test_forecast_wrapper.py` | ForecastObsWrapper observation space augmentation |
| `test_temporal_wrapper.py` | TemporalHistoryWrapper stacking and index mapping |
| `test_action_clamp.py` | Battery and EV action clamping bounds |
| `test_schema_index.py` | ObsIndex construction for 5-building schema |
| `test_saute_wrapper.py` | Saute budget wrapper observation augmentation and penalty |
| `test_config_loading.py` | YAML deep-merge with OmniSafe defaults |

## Notes

- Tests require the `citylearn` conda environment and the 5-building schema at
  `data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json`.
- Some tests instantiate the full CityLearn environment, which takes a few
  seconds per test. Use `pytest -x` to stop on first failure during debugging.
- GPU is not required; tests run on CPU by default.
