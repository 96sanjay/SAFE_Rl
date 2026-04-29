# Diagnostic Scripts

One-off scripts written during development to investigate specific bugs, validate assumptions, or analyze intermediate results. These are not part of the test suite and are not maintained. They are preserved for reproducibility -- if a question arises about how a specific issue was diagnosed, the script that answered it is here.

## Categories

**`debug_*.py`** -- Debugging specific issues. Written to isolate a bug, run once or twice, then kept for reference. Examples: debugging observation shapes, gradient flow, action scaling.

**`diagnose_*.py`** -- Constraint cost analysis. Scripts that break down where constraint costs come from (which buildings, which timesteps, which constraint channels).

**`check_*.py` / `verify_*.py`** -- Validation checks. Confirm that an environment wrapper, reward function, or config change behaves as expected. Typically run after a code change to sanity-check before launching a full training run.

**`test_*.py`** -- Ad-hoc tests. NOT part of the project's formal test suite. Quick scripts to test a hypothesis or verify a fix in isolation.

**`analyze_*.py`** -- One-off analysis. Scripts that extract and visualize specific metrics from training logs. Often written to generate a single plot or table for a decision point.

## Usage

These scripts may have hardcoded paths, missing dependencies, or assumptions about the environment state at the time they were written. They are reference material, not runnable tools. If you need to reuse one, expect to update paths and imports.
