# configs/

YAML configuration files for all training runs.

## Directory Structure

```
configs/
  active/           Currently used configurations
    r27a_cmdp.yaml      R27a headroom-gated CMDP (PPOLagMulti, 5-building)
    r28_*.yaml          R28 OmniSafe stock benchmark configs
  archive/          Historical configurations from previous runs
    on-policy/          PPOLag, PPOLagMulti, TRPOLag variants
    off-policy/         SACLag, CSAC-LB variants
```

## Naming Convention

```
r{run_number}{variant}_{description}.yaml
```

- **run_number**: Sequential experiment ID (e.g., 27)
- **variant**: Letter suffix for sub-experiments (e.g., `a`, `b`, `j`)
- **description**: Short label (e.g., `cmdp`, `stems`, `lagfix`)

Examples:
- `r27a_cmdp.yaml` -- Run 27, variant a, CMDP configuration
- `r26j_lagfix.yaml` -- Run 26, variant j, Lagrangian fix experiment

## Configuration Reference

See [docs/CONFIGURATION_GUIDE.md](../docs/CONFIGURATION_GUIDE.md) for complete
documentation of all YAML keys, environment variables, CLI arguments, and PID
Lagrangian tuning.
