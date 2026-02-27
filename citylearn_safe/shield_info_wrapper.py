"""
ShieldInfoWrapper: guarantees info["shielded_action"] exists after step().
Wraps ActionProjectionWrapper transparently.
"""
import numpy as np
import gymnasium as gym


class ShieldInfoWrapper(gym.Wrapper):
    def step(self, action):
        raw = np.asarray(action, dtype=np.float32).flatten().copy()
        obs, reward, term, trunc, info = self.env.step(action)

        if "shielded_action" not in info:
            # Try alternate keys the AP wrapper might use
            for key in ("projected_action", "ap_projected_action",
                        "safe_action", "corrected_action"):
                if key in info:
                    info["shielded_action"] = np.asarray(
                        info[key], dtype=np.float32).flatten()
                    break
            else:
                # Check if delta indicates shield was active
                delta = float(info.get("ap_action_delta_l2", 0.0))
                if delta > 1e-8:
                    info["shielded_action"] = raw  # can't recover, flag it
                    info["_shield_action_missing"] = True
                else:
                    info["shielded_action"] = raw  # no intervention

        return obs, reward, term, trunc, info

    def reset(self, **kw):
        return self.env.reset(**kw)
