from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from citylearn_safe.diff_projector import DiffProjector
from citylearn_safe.lookahead_psf_v2g import LookaheadPSFv2G


class HybridProjector:
    """Hybrid hard shield for SP-RL execution.

    Design:
      - Use PSF-style lookahead/MPC only to derive live EV feasibility bounds.
      - Merge those bounds with DiffProjector's own EV guards.
      - Solve one final joint repair through DiffProjector for C0/C2/C3/C4.

    During replay-buffer batch projection (state_tensors is provided), we fall
    back to DiffProjector-only projection because the exact PSF horizon state is
    not stored per sample. Execution-time shielding remains the primary safety
    mechanism.
    """

    def __init__(
        self,
        env: Any,
        horizon: int = 24,
        solver_eps: float = 1e-4,
        solver_max_iters: int = 5000,
        verbose: int = 0,
    ) -> None:
        self.env = env
        self.verbose = int(verbose)
        self._diff = DiffProjector(
            env=env,
            solver_eps=solver_eps,
            solver_max_iters=solver_max_iters,
        )
        self._psf = LookaheadPSFv2G(
            env=env,
            horizon=horizon,
            verbose=verbose,
        )
        self._built = False

    @property
    def state_tensor_dim(self) -> int:
        return self._diff.state_tensor_dim

    def build(self) -> None:
        self._diff.build()
        if not getattr(self._psf, "_compiled", False):
            self._psf._compile_qp()
        self._built = True

    def extract_state_tensor(self) -> torch.Tensor:
        if not self._built:
            self.build()
        return self._diff.extract_state_tensor()

    def _bounded_ev_state_tensor(
        self,
        diff_state: torch.Tensor,
        ev_min: np.ndarray,
        ev_max: np.ndarray,
    ) -> torch.Tensor:
        """Replace EV min/max bounds in a DiffProjector state tensor."""
        if not self._built:
            self.build()
        m = self._diff._mapping
        nb = m["nb"]
        n_batt = len(m["batt_gidx"])
        n_ev = len(m["ev_gidx"])
        if n_ev == 0:
            return diff_state

        out = diff_state.clone()
        o1 = nb
        o2 = o1 + n_batt
        o3 = o2 + n_batt
        o4 = o3 + n_batt
        o5 = o4 + n_batt
        o6 = o5 + n_ev
        o7 = o6 + n_ev

        ev_min_t = torch.as_tensor(ev_min, dtype=out.dtype, device=out.device)
        ev_max_t = torch.as_tensor(ev_max, dtype=out.dtype, device=out.device)
        out[..., o6:o7] = ev_min_t
        out[..., o7:o7 + n_ev] = ev_max_t
        return out

    def _psf_ev_bounds(
        self,
        proposed_flat: np.ndarray,
        base_ev_min: np.ndarray,
        base_ev_max: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Use PSF lookahead only to derive live EV feasibility bounds."""
        _corrected_flat, psf_info = self._psf._update_params_and_solve(proposed_flat)

        n_ev = len(self._psf._mapping["ev_chargers"])
        ev_min = np.asarray(base_ev_min, dtype=np.float32).copy()
        ev_max = np.asarray(base_ev_max, dtype=np.float32).copy()

        p_ev_prefix_min = getattr(self._psf, "_p_ev_prefix_min", None)
        p_ev_c1_coeff = getattr(self._psf, "_p_ev_c1_coeff", None)
        if p_ev_prefix_min is not None and p_ev_c1_coeff is not None:
            prefix = np.asarray(p_ev_prefix_min.value, dtype=float) if p_ev_prefix_min.value is not None else None
            coeff = np.asarray(p_ev_c1_coeff.value, dtype=float) if p_ev_c1_coeff.value is not None else None
            if prefix is not None and coeff is not None:
                for ei in range(min(n_ev, prefix.shape[0], coeff.shape[0])):
                    c0 = float(coeff[ei, 0]) if coeff.ndim == 2 and coeff.shape[1] > 0 else 0.0
                    p0 = float(prefix[ei, 0]) if prefix.ndim == 2 and prefix.shape[1] > 0 else 0.0
                    if c0 > 1e-9 and p0 > 0.0:
                        ev_min[ei] = max(ev_min[ei], float(np.clip(p0 / c0, ev_min[ei], ev_max[ei])))

        return ev_min, ev_max, psf_info

    def _project_live(
        self,
        obs_tensor: torch.Tensor,
        unsafe_action: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Execution-time shield: PSF EV bounds + one joint DiffProjector repair."""
        squeeze_output = False
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        if unsafe_action.dim() == 1:
            unsafe_action = unsafe_action.unsqueeze(0)
            squeeze_output = True
        if unsafe_action.dim() != 2 or unsafe_action.shape[0] != 1:
            return self._diff.project(obs_tensor, unsafe_action)

        proposed_flat = unsafe_action[0].detach().cpu().numpy().astype(float, copy=True)
        diff_state = self._diff.extract_state_tensor().unsqueeze(0).to(
            device=unsafe_action.device,
            dtype=unsafe_action.dtype,
        )
        unpacked = self._diff._unpack_state_tensor(diff_state)
        base_ev_min = unpacked[6][0].detach().cpu().numpy()
        base_ev_max = unpacked[7][0].detach().cpu().numpy()
        ev_min, ev_max, psf_info = self._psf_ev_bounds(proposed_flat, base_ev_min, base_ev_max)
        fixed_state = self._bounded_ev_state_tensor(diff_state, ev_min, ev_max)
        act_safe, diff_info = self._diff.project(
            obs_tensor,
            unsafe_action,
            state_tensors=fixed_state,
        )
        info = dict(diff_info)
        info["hybrid_used_psf_c0"] = True
        info["hybrid_psf_status"] = psf_info.get("psf_status", "unknown")
        info["hybrid_psf_solver"] = psf_info.get("psf_solver", "unknown")
        info["hybrid_psf_infeasible"] = psf_info.get("psf_infeasible", 0.0)
        if squeeze_output:
            act_safe = act_safe.squeeze(0)
        return act_safe, info

    def project(
        self,
        obs_tensor: torch.Tensor,
        unsafe_action: torch.Tensor,
        state_tensors: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if not self._built:
            self.build()

        # Replay/batch path: use DiffProjector on stored sample states.
        if state_tensors is not None:
            act_safe, info = self._diff.project(
                obs_tensor,
                unsafe_action,
                state_tensors=state_tensors,
            )
            info["hybrid_used_psf_c0"] = False
            return act_safe, info

        return self._project_live(obs_tensor, unsafe_action)
