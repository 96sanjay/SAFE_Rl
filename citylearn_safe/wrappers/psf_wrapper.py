from __future__ import annotations
import os
import gymnasium as gym

from citylearn_safe.psf.psf_filter import PredictiveSafetyFilter, PSFConfig

class PSFShieldWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        cfg = PSFConfig(
            horizon=int(os.environ.get("CITYLEARN_PSF_HORIZON", "24")),
            soc_low=float(os.environ.get("CITYLEARN_STEMS_SOC_LOW", "0.0")),
            soc_high=float(os.environ.get("CITYLEARN_STEMS_SOC_HIGH", "0.95")),
            p_building_max=float(os.environ.get("CITYLEARN_STEMS_P_BUILDING_MAX", "2.273834")),
            p_grid_max=float(os.environ.get("CITYLEARN_STEMS_P_GRID_MAX", "27.127751")),
        )
        self.psf = PredictiveSafetyFilter(env, cfg)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.psf.reset()
        info = dict(info)
        info.update({
            "psf_active": 0.0,
            "psf_status": "reset",
            "psf_delta_l2": 0.0,
            "psf_solve_ms": 0.0,
            "psf_used_fallback": 0.0,
            "psf_infeasible": 0.0,
            "psf_ev_active": 0.0,
            "psf_horizon": float(self.psf.cfg.horizon),
        })
        return obs, info

    def step(self, action):
        safe_action, psf_info = self.psf.filter(action)
        obs, reward, term, trunc, info = self.env.step(safe_action)
        info = dict(info)
        info.update(psf_info)
        return obs, reward, term, trunc, info
