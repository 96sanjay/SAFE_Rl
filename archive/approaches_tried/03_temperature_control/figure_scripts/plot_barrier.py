#!/usr/bin/env python3
"""Plot the linear smoothed log barrier function for thesis."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11, "font.family": "serif",
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "text.usetex": False,
})

x = np.linspace(-3.0, 2.0, 1000)

def smoothed_barrier(x, mu):
    threshold = -1.0 / (mu**2)
    out = np.empty_like(x)
    log_mask = x <= threshold
    out[log_mask] = -(1.0/mu) * np.log(-x[log_mask])
    lin_mask = ~log_mask
    const = np.log(1.0 / (mu**2))
    out[lin_mask] = mu * x[lin_mask] - (1.0/mu) * const + (1.0/mu)
    return out

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: barrier function for different mu
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
for mu_val, color in zip([1, 5, 10, 25], colors):
    y = smoothed_barrier(x, mu_val)
    ax1.plot(x, y, linewidth=2, label=r"$\mu = %d$" % mu_val, color=color)

# Indicator function (dashed)
ax1.plot(x[x<=0], np.zeros_like(x[x<=0]), "k--", linewidth=1.5, label="Indicator")
ax1.plot([0, 0], [0, 20], "k--", linewidth=1.5)
ax1.plot(x[x>0], np.ones_like(x[x>0])*20, "k--", linewidth=1.5)

ax1.set_xlim(-3, 2)
ax1.set_ylim(-5, 25)
ax1.set_xlabel("x")
ax1.set_ylabel(r"$\psi(x)$")
ax1.set_title("Linear Smoothed Log Barrier Function")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Right: shifted barrier with ReLU (as used in CSAC-LB)
d = 0.5
mu = 25.0
qc = np.linspace(-1.0, 3.0, 1000)
shift = 1.0 / (mu**2)
slack = np.maximum(qc - d, 0.0) - shift
penalty = smoothed_barrier(slack, mu)

ax2.plot(qc, penalty, linewidth=2, color="#d62728",
         label=r"Barrier penalty, $\mu=25$")
ax2.axvline(x=d, color="gray", linestyle="--", linewidth=1,
            label=r"Cost limit $d=0.5$")
ax2.fill_betweenx([penalty.min()-1, penalty.max()+1], d, qc.max(),
                   alpha=0.05, color="red")
ax2.annotate("Infeasible\nregion", xy=(d+0.5, penalty.max()*0.3),
             fontsize=10, color="red", ha="center")

ax2.set_xlabel(r"$Q_C(s, a)$")
ax2.set_ylabel("Barrier Penalty")
ax2.set_title(r"CSAC-LB Shifted Barrier: $\psi^*(ReLU(Q_C - d) - 1/\mu^2)$")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(penalty.min()-0.5, min(penalty.max(), 30))

fig.tight_layout()
fig.savefig("/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/docs/temperature_case_study/figures/fig_barrier_function.pdf")
fig.savefig("/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/docs/temperature_case_study/figures/fig_barrier_function.png")
print("Barrier function plot saved.")
