#!/usr/bin/env python3
"""
Plot episode-level reward diagnostics for a training run.

Usage:
    python scripts/plot_episode_rewards.py <run_directory>
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main(run_dir: str) -> None:
    run_path = Path(run_dir)
    progress_path = run_path / "progress.csv"
    kpi_path = run_path / "kpis_kpis.csv"

    if not progress_path.exists() or not kpi_path.exists():
        print(f"Missing required files in {run_dir}")
        return

    progress = pd.read_csv(progress_path)
    kpis = pd.read_csv(kpi_path)

    # Raw environment reward: average per episode using step-wise data (exclude summary rows with step == 0)
    step_rows = kpis[kpis["step"] > 0]
    raw_episode_mean = step_rows.groupby("episode")["reward"].mean()

    # OmniSafe metrics: normalized episode return & per-step average using logged episode length
    if "Metrics/EpRet" in progress.columns and "Metrics/EpLen" in progress.columns:
        ep_ret = progress["Metrics/EpRet"]
        ep_len = progress["Metrics/EpLen"].replace(0, np.nan)
        normalized_episode_mean = ep_ret / ep_len
    else:
        print("Warning: Metrics/EpRet or Metrics/EpLen missing in progress.csv")
        normalized_episode_mean = pd.Series(dtype=float)

    # Align indices
    episodes = sorted(set(raw_episode_mean.index).intersection(normalized_episode_mean.index))
    if not episodes:
        print("No overlapping episodes to plot.")
        return

    raw_values = raw_episode_mean.loc[episodes]
    norm_values = normalized_episode_mean.loc[episodes]

    plt.figure(figsize=(12, 6))
    plt.plot(episodes, raw_values, label="Raw Avg Reward (per episode)", color="tab:blue")
    plt.plot(episodes, norm_values, label="OmniSafe Avg Reward (EpRet/EpLen)", color="tab:orange")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title(f"Episode Reward Comparison - {run_path.name}")
    plt.grid(True, alpha=0.3)
    plt.legend()

    output_file = run_path / "episode_reward_compare.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to: {output_file}")

    # Print summary statistics
    diff = raw_values - norm_values
    print("\n=== SUMMARY ===")
    print(f"Episodes compared: {len(episodes)}")
    print(f"Raw avg reward (latest episode): {raw_values.iloc[-1]:.3f}")
    print(f"OmniSafe avg reward (latest episode): {norm_values.iloc[-1]:.3f}")
    print(f"Mean absolute difference: {diff.abs().mean():.3f}")
    print(f"Max absolute difference: {diff.abs().max():.3f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/plot_episode_rewards.py <run_directory>")
        sys.exit(1)
    main(sys.argv[1])

