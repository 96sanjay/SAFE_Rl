# citylearn_safe/adapters.py
#Without this adapter, OmniSafe would crash because it expects gym.spaces.Box, not List[Box].



# CityLearn's central_agent=True still returns:
#observation_space = [Box(...)]  # Length-1 list!
#action_space = [Box(...)]       # Length-1 list!

# Adapter unwraps this to plain Box:
#observation_space = Box(...)    # What OmniSafe expects
#action_space = Box(...)         # What OmniSafe expects



from __future__ import annotations
import numpy as np
import gymnasium as gym
from typing import Any, Dict, Tuple, List

def _first(x):
    return x[0] if isinstance(x, (list, tuple)) else x

class SingleAgentListAdapter(gym.Env):
    """
    Wraps a CityLearn env that exposes single-agent spaces as length-1 lists.
    Converts obs/actions to plain Box and (obs, reward, terminated, truncated, info) scalars.
    """
    metadata = {"render_modes": []}

    def __init__(self, base_env: Any):
        super().__init__()
        self.base = base_env

        # Unwrap list-wrapped spaces (len==1)
        obs_space = getattr(self.base, "observation_space", None)
        act_space = getattr(self.base, "action_space", None)
        if isinstance(obs_space, (list, tuple)):
            assert len(obs_space) == 1, "Expected single-agent list of length 1."
            obs_space = obs_space[0]
        if isinstance(act_space, (list, tuple)):
            assert len(act_space) == 1, "Expected single-agent list of length 1."
            act_space = act_space[0]
        self.observation_space = obs_space
        self.action_space = act_space

    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        obs, info = self.base.reset(seed=seed, options=options)
        obs = _first(obs)
        return np.asarray(obs, dtype=np.float32), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Base env expects a list [action] even for single agent
        a = [np.asarray(action, dtype=np.float32)] if isinstance(self.base.action_space, (list, tuple)) else action
        obs, r, term, trunc, info = self.base.step(a)
        obs = _first(obs)
        r = float(_first(r))
        term = bool(_first(term))
        trunc = bool(_first(trunc))
        return np.asarray(obs, dtype=np.float32), r, term, trunc, info

    def render(self):
        return getattr(self.base, "render", lambda: None)()

    def close(self):
        return getattr(self.base, "close", lambda: None)()




