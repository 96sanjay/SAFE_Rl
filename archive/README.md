# Archive

This directory preserves every abandoned approach, superseded script, and one-off diagnostic tool produced during the Safe RL V2G project. Nothing is deleted. Failed experiments are as valuable as successful ones -- they document what does not work, why, and what insight each failure contributed to the final per-channel CMDP pipeline.

## Why This Archive Exists

Three reasons:

1. **Scientific rigor.** The thesis claims that the per-channel CMDP pipeline (PPO-Lag-Multi) outperforms alternatives. Those alternatives must be inspectable to verify the claim.
2. **Reproducibility.** Every experiment in the run lineage (R11b through R27a) can be traced back to its code, even experiments that were stopped early or abandoned.
3. **Knowledge transfer.** Future researchers on this codebase should know which directions were explored and why they were closed, without repeating dead-end work.

## Directory Structure

| Directory | Contents | Purpose |
|-----------|----------|---------|
| `approaches_tried/` | 6 abandoned approaches + 1 completed auxiliary case study | Full record of every major design direction explored before arriving at the per-channel CMDP pipeline |
| `diagnostic_scripts/` | ~50 one-off debug, diagnosis, and validation scripts | Ad-hoc tools written to investigate specific bugs or behaviors during development |
| `evaluation_scripts/` | Superseded evaluation scripts (two subdirs: `root/`, `scripts_dir/`) | Evaluation code replaced by the active scripts in `/scripts/evaluation/` |
| `training_experiments/` | Experimental training scripts (two subdirs: `root/`, `scripts_dir/`) | Training code replaced by the active scripts in `/scripts/training/` |

## How to Navigate

**Looking for why an approach was abandoned?** Start with `approaches_tried/README.md`. Each of the 7 entries documents the idea, the files, the results, and the lesson learned.

**Looking for a specific old script?** Check `diagnostic_scripts/`, `evaluation_scripts/`, or `training_experiments/` depending on the script's purpose. Each has its own README with a categorized listing.

**Looking for active code?** This archive contains only inactive code. Active scripts live in:
- `/scripts/training/` -- current training pipelines
- `/scripts/evaluation/` -- current evaluation pipelines
- `/envs/` -- current environment wrappers

## Relationship to Run Lineage

The project's run lineage is:

```
R11b -> R12a -> R15a-d -> R18 -> R25b (baseline)
R26: a-c (ablation) -> d (MLP) -> e (STEMS rich) -> f (minimal) -> g (critic fix)
     -> h/i (forecast arb) -> j (lagfix) -> R27a (CMDP, current)
```

Each archived approach maps to a specific era in this lineage. The `approaches_tried/README.md` provides the mapping.
