#!/usr/bin/env python3
"""
Statistical significance analysis for multi-seed evaluation results.

Computes:
  - 95% bootstrap confidence intervals for mean violation rate
  - Pairwise Mann-Whitney U tests (CSAC-LB vs baselines, SAC-Lag vs CPO)
  - LaTeX table snippet for thesis

Caveat: With n=3--6 seeds, statistical power is inherently limited.
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path
from itertools import combinations

# ── Configuration ──────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).resolve().parents[3] / "runs" / "multiseed_eval_summary.json"
N_BOOTSTRAP = 10_000
CI_LEVEL = 0.95
RNG_SEED = 42

PAIRWISE_COMPARISONS = [
    ("CSAC-LB", "SAC-Lag"),
    ("CSAC-LB", "CPO"),
    ("CSAC-LB", "FOCOPS"),
    ("SAC-Lag", "CPO"),
]


def load_violation_rates(path: Path) -> dict[str, np.ndarray]:
    """Load per-seed violation rates from the summary JSON."""
    with open(path) as f:
        data = json.load(f)
    return {
        algo: np.array([s["violation_rate"] for s in info["per_seed"]])
        for algo, info in data.items()
    }


def bootstrap_ci(samples: np.ndarray, n_boot: int, ci: float, rng: np.random.Generator) -> tuple[float, float, float]:
    """Return (mean, ci_lo, ci_hi) via percentile bootstrap."""
    n = len(samples)
    boot_means = np.array([
        rng.choice(samples, size=n, replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_means, [alpha, 1 - alpha])
    return float(samples.mean()), float(lo), float(hi)


def mann_whitney_test(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Two-sided Mann-Whitney U test. Returns (U statistic, p-value)."""
    result = stats.mannwhitneyu(x, y, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def rank_biserial_r(u: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation as effect size for Mann-Whitney U."""
    return 1 - (2 * u) / (n1 * n2)


def main():
    rng = np.random.default_rng(RNG_SEED)
    rates = load_violation_rates(DATA_PATH)

    # Desired display order
    algo_order = ["CSAC-LB", "SAC-Lag", "CPO", "CUP", "FOCOPS"]

    # ── Bootstrap CIs ─────────────────────────────────────────────────────
    ci_results = {}
    for algo in algo_order:
        v = rates[algo]
        mean, lo, hi = bootstrap_ci(v, N_BOOTSTRAP, CI_LEVEL, rng)
        ci_results[algo] = (mean, lo, hi, len(v))

    print("=" * 72)
    print("Bootstrap 95% Confidence Intervals for Mean Violation Rate")
    print(f"(B = {N_BOOTSTRAP} resamples, percentile method)")
    print("=" * 72)
    print(f"{'Algorithm':<12} {'n':>3}  {'Mean':>7}  {'95% CI':>19}")
    print("-" * 50)
    for algo in algo_order:
        mean, lo, hi, n = ci_results[algo]
        print(f"{algo:<12} {n:>3}  {mean:>7.3f}  [{lo:.3f}, {hi:.3f}]")

    # ── Pairwise Mann-Whitney U ────────────────────────────────────────────
    print()
    print("=" * 72)
    print("Pairwise Mann-Whitney U Tests (two-sided)")
    print("=" * 72)
    print(f"{'Comparison':<22} {'U':>6}  {'p-value':>8}  {'r_rb':>6}  {'Sig.':>5}")
    print("-" * 55)

    mw_results = []
    for a, b in PAIRWISE_COMPARISONS:
        u, p = mann_whitney_test(rates[a], rates[b])
        r = rank_biserial_r(u, len(rates[a]), len(rates[b]))
        sig = "* " if p < 0.05 else "ns"
        label = f"{a} vs {b}"
        print(f"{label:<22} {u:>6.1f}  {p:>8.4f}  {r:>+6.3f}  {sig:>5}")
        mw_results.append((a, b, u, p, r))

    # ── Caveat ─────────────────────────────────────────────────────────────
    print()
    print("-" * 72)
    print("CAVEAT: Sample sizes are very small (n = 3--6). Statistical power")
    print("is limited; non-significant results should NOT be interpreted as")
    print("evidence of no difference. Effect sizes (rank-biserial r) are")
    print("reported alongside p-values to aid interpretation.")
    print("The minimum achievable p-value for n1=6 vs n2=3 is ~0.024;")
    print("for n1=3 vs n2=3 it is 0.100 (U test cannot reach p<0.05).")
    print("-" * 72)

    # ── LaTeX Table ────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("LaTeX Table Snippet")
    print("=" * 72)
    print()

    # Table 1: Bootstrap CIs
    print(r"\begin{table}[htbp]")
    print(r"  \centering")
    print(r"  \caption{Mean violation rate with 95\% bootstrap confidence intervals"
          r" ($B = 10{,}000$). Sample sizes are small ($n = 3$--$6$),"
          r" so intervals should be interpreted cautiously.}")
    print(r"  \label{tab:violation-rate-ci}")
    print(r"  \begin{tabular}{l c c c}")
    print(r"    \toprule")
    print(r"    Algorithm & $n$ & Mean VR & 95\% CI \\")
    print(r"    \midrule")
    for algo in algo_order:
        mean, lo, hi, n = ci_results[algo]
        # Bold the best (lowest) mean
        bold = algo == "CSAC-LB"
        mean_str = f"\\textbf{{{mean:.3f}}}" if bold else f"{mean:.3f}"
        ci_str = f"[{lo:.3f},\\,{hi:.3f}]"
        if bold:
            ci_str = f"\\textbf{{{ci_str}}}"
        print(f"    {algo} & {n} & {mean_str} & {ci_str} \\\\")
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")

    print()

    # Table 2: Pairwise tests
    print(r"\begin{table}[htbp]")
    print(r"  \centering")
    print(r"  \caption{Pairwise Mann--Whitney $U$ tests (two-sided)."
          r" With $n \leq 6$, power is limited;"
          r" $r_{\mathrm{rb}}$ = rank-biserial effect size.}")
    print(r"  \label{tab:pairwise-mwu}")
    print(r"  \begin{tabular}{l c c c}")
    print(r"    \toprule")
    print(r"    Comparison & $U$ & $p$ & $r_{\mathrm{rb}}$ \\")
    print(r"    \midrule")
    for a, b, u, p, r in mw_results:
        p_str = f"{p:.3f}" if p >= 0.001 else f"$<$0.001"
        sig_marker = "$^{*}$" if p < 0.05 else ""
        print(f"    {a} vs.\\ {b} & {u:.1f} & {p_str}{sig_marker} & {r:+.3f} \\\\")
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
