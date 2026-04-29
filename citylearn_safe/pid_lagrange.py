"""PID Lagrangian with cost-limit normalization.

Implements the PID Lagrangian method from Stooke, Achiam, Abbeel (ICML 2020,
arxiv 2007.03964) with an additional cost-limit normalization step so that
a single set of PID gains works across constraints with very different
cost magnitudes (e.g., C0 gap ~154 vs C3 gap ~22,000).

Normalization: delta = (ep_cost - cost_limit) / cost_limit

This makes delta dimensionless and O(1) for all constraints, so the default
gains (Kp=0.1, Ki=0.01, Kd=0.01) produce reasonable lambda dynamics without
per-constraint tuning.

Drop-in replacement for omnisafe.common.lagrange.Lagrange — exposes the same
.lagrangian_multiplier property.
"""
from __future__ import annotations

from collections import deque


class PIDLagrange:
    """PID controller for Lagrangian multiplier with cost-limit normalization.

    Instead of raw SGD: lambda += lr * (ep_cost - limit)
    Uses PID:          lambda = Kp * error_p + Ki * integral + Kd * derivative
    where error = (ep_cost - limit) / limit  (normalized).

    This prevents integral windup that causes StopIter oscillation in standard
    Lagrangian methods.

    Args:
        cost_limit: The constraint threshold d_k.
        pid_kp: Proportional gain. Reacts to current violation magnitude.
        pid_ki: Integral gain. Accumulates persistent violations.
        pid_kd: Derivative gain. Dampens oscillation when cost is changing.
        pid_d_delay: Epochs to look back for D-term comparison.
        pid_delta_p_ema_alpha: EMA smoothing for P-term (0.95 = heavy smoothing).
        pid_delta_d_ema_alpha: EMA smoothing for D-term.
        penalty_max: Upper bound on lambda (like lagrangian_upper_bound).
        lagrangian_multiplier_init: Initial I-term value.
        normalize_by_limit: If True, normalize delta by cost_limit.
    """

    def __init__(
        self,
        cost_limit: float,
        pid_kp: float = 0.1,
        pid_ki: float = 0.01,
        pid_kd: float = 0.01,
        pid_d_delay: int = 10,
        pid_delta_p_ema_alpha: float = 0.95,
        pid_delta_d_ema_alpha: float = 0.95,
        penalty_max: float = 3.0,
        lagrangian_multiplier_init: float = 0.001,
        normalize_by_limit: bool = True,
    ) -> None:
        self._cost_limit = cost_limit
        self._pid_kp = pid_kp
        self._pid_ki = pid_ki
        self._pid_kd = pid_kd
        self._pid_d_delay = pid_d_delay
        self._ema_alpha_p = pid_delta_p_ema_alpha
        self._ema_alpha_d = pid_delta_d_ema_alpha
        self._penalty_max = penalty_max
        self._normalize = normalize_by_limit

        # PID state
        self._pid_i: float = lagrangian_multiplier_init   # I-term (accumulated)
        self._delta_p: float = 0.0                        # P-term (EMA-smoothed error)
        self._cost_d: float = 0.0                         # D-term (EMA-smoothed cost)
        self._cost_ds: deque[float] = deque(maxlen=max(1, pid_d_delay))
        self._cost_ds.append(0.0)

        # Output
        self._cost_penalty: float = 0.0

    def update_cost_limit(self, new_limit: float) -> None:
        """Update cost limit for curriculum annealing (R18).

        Rescales integral term proportionally to prevent windup from
        stale accumulation at different cost-limit scales.
        """
        old_limit = self._cost_limit
        self._cost_limit = new_limit

        if self._normalize and old_limit > 0 and new_limit > 0:
            scale = old_limit / new_limit
            self._pid_i *= scale
        elif new_limit <= 0:
            self._pid_i = 0.0

    @property
    def lagrangian_multiplier(self) -> float:
        """The current lambda value. Compatible with Lagrange interface."""
        return self._cost_penalty

    def pid_update(self, ep_cost_avg: float) -> None:
        """Update PID controller with this epoch's cost.

        Args:
            ep_cost_avg: The episode cost for this constraint this epoch.
        """
        # Raw error
        raw_delta = float(ep_cost_avg) - self._cost_limit

        # Normalize by cost_limit for scale-invariant gains
        if self._normalize and self._cost_limit > 0:
            delta = raw_delta / self._cost_limit
            cost_for_d = float(ep_cost_avg) / self._cost_limit
        else:
            delta = raw_delta
            cost_for_d = float(ep_cost_avg)

        # I-term: accumulated error (like standard Lagrangian, but slower)
        # R19: Anti-windup — clamp I-term to penalty_max to prevent accumulation
        # when output is saturated. Without this, I-term grows unboundedly while
        # lambda is capped, causing delayed response when costs finally drop.
        self._pid_i = max(0.0, self._pid_i + delta * self._pid_ki)
        self._pid_i = min(self._pid_i, self._penalty_max)

        # P-term: EMA-smoothed current error (immediate response, no accumulation)
        self._delta_p *= self._ema_alpha_p
        self._delta_p += (1.0 - self._ema_alpha_p) * delta

        # D-term: EMA-smoothed cost derivative (brakes when cost is decreasing)
        self._cost_d *= self._ema_alpha_d
        self._cost_d += (1.0 - self._ema_alpha_d) * cost_for_d
        pid_d = max(0.0, self._cost_d - self._cost_ds[0])

        # PID output
        pid_o = self._pid_kp * self._delta_p + self._pid_i + self._pid_kd * pid_d

        # Clamp to [0, penalty_max]
        self._cost_penalty = max(0.0, min(pid_o, self._penalty_max))

        # Store smoothed cost for future D-term computation
        self._cost_ds.append(self._cost_d)

    def __repr__(self) -> str:
        return (
            f"PIDLagrange(limit={self._cost_limit}, "
            f"lambda={self._cost_penalty:.4f}, "
            f"I={self._pid_i:.4f}, P={self._delta_p:.4f})"
        )
