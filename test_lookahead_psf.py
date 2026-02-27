import os, sys, time
os.chdir("/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork")
sys.path.insert(0, os.getcwd())
os.environ["CITYLEARN_STEMS_SOC_LOW"] = "0.0"
os.environ["CITYLEARN_STEMS_SOC_HIGH"] = "0.95"
os.environ["CITYLEARN_STEMS_P_BUILDING_MAX"] = "3.47"
os.environ["CITYLEARN_STEMS_P_GRID_MAX"] = "35.76"
os.environ["CITYLEARN_EV_MISSING_ACTION_MODE"] = "assume_zero"

import json, numpy as np
from citylearn.citylearn import CityLearnEnv
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.psf.psf_lookahead import LookaheadPSFWrapper

schema_path = "data/citylearn_challenge_2022_phase_all_plus_evs/schema.json"
with open(schema_path) as f:
    schema = json.load(f)
schema["root_directory"] = os.path.dirname(os.path.abspath(schema_path))

base_env = CityLearnEnv(schema=schema)
safety_env = CityLearnSafetyEnvV3(base_env)
env = LookaheadPSFWrapper(safety_env, horizon=12, verbose=1)

obs, info = env.reset()
print("Quick test: 50 steps with zero actions")
t0 = time.time()
for i in range(50):
    action = [np.zeros(sp.shape, dtype=np.float32) for sp in env.action_space]
    obs, reward, term, trunc, info = env.step(action)
    if i < 3:
        print("  step %d: status=%s solve=%.1fms delta=%.4f" % (
            i, info.get("psf_status", "?"),
            info.get("psf_solve_ms", 0),
            info.get("psf_action_delta_l2", 0)))
elapsed = time.time() - t0
print("50 steps in %.1fs (%.0fms/step)" % (elapsed, elapsed*1000/50))
