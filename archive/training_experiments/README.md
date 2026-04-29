# Training Experiments (Archived)

Experimental training scripts from various project phases. These were one-off or short-lived training configurations that have been superseded by the active training pipeline in `/scripts/training/`.

## Subdirectories

**`root/`** -- One-off training experiments that originally lived in the project root. These were typically quick-turnaround scripts for testing a hypothesis (new reward weight, different network size, alternate constraint formulation) before committing to a full run.

**`scripts_dir/`** -- Superseded training scripts that originally lived in the `scripts/` directory. These represent earlier iterations of the training pipeline that were replaced as the environment wrappers, reward functions, and constraint definitions evolved through the R11-R27 run lineage.

## Active Training Code

Do not use the scripts in this archive for new training runs. The active training scripts are in:

```
/scripts/training/
```

These archived scripts may reference old config formats, deprecated environment wrappers, or outdated hyperparameter conventions.
