export CITYLEARN_SCHEMA="$PWD/data/citylearn/schema.json"
python -m scripts.register_env
python -m scripts.train_omnisafe --cfg configs/ppo_lag_soc.yaml




safe-citylearn % python - <<'PY'
import gymnasium as gym
import scripts.register_env  # registers Simple-v0 -> your thunk
from citylearn_safe.safety_env import CityLearnSafetyEnv
e = gym.make("Simple-v0")
print("Spec:", e.spec.id, "| Type:", type(e.unwrapped).__name__)
assert isinstance(e.unwrapped, CityLearnSafetyEnv), "Not our env!"
print("OK: using CityLearnSafetyEnv via 'Simple-v0'")
PY

Couldn't import dot_parser, loading of dot files will not be possible.
Spec: Simple-v0 | Type: CityLearnSafetyEnv
OK: using CityLearnSafetyEnv via 'Simple-v0




#Test the algo imported is correct

python - <<'PY'
import yaml, omnisafe, citylearn_safe.omni_env
cfg = yaml.safe_load(open("configs/ppo_lag_soc.yaml"))
agent = omnisafe.Agent(cfg["algo"], cfg["env_id"], custom_cfgs=cfg)
print("Algo class:", agent.agent.__class__.__name__)
PY



('NaturalPG', 'PolicyGradient', 'PPO', 'TRPO', 'TRPOEarlyTerminated', 'PPOEarlyTerminated', 'CUP', 'FOCOPS', 'RCPO', 'PDO', 'PPOLag', 'TRPOLag', 'OnCRPO', 'P3O', 'IPO', 'CPPOPID', 'TRPOPID', 'TRPOSaute', 'PPOSaute', 'CPO', 'PCPO', 'TRPOSimmerPID', 'PPOSimmerPID', 'DDPG', 'TD3', 'SAC', 'DDPGLag', 'TD3Lag', 'SACLag', 'DDPGPID', 'TD3PID', 'SACPID', 'LOOP', 'PETS', 'CAPPETS', 'CCEPETS', 'SafeLOOP', 'RCEPETS', 'BCQ', 'BCQLag', 'CCRR', 'CRR', 'COptiDICE', 'VAEBC').





#RUN

export CITYLEARN_SCHEMA="$PWD/data/citylearn/schema.json"
python -m scripts.benchmark_algos



Central agent mode transforms a complex 17-agent MARL problem into a single-agent CMDP that OmniSafe can handle natively. The constraint becomes a collective safety requirement where the policy must keep ALL buildings' batteries within safe bounds simultaneously.


# ========================================
# 1-BUILDING SETUP (for faster testing)
# ========================================

# The schema is now configured for 1 building only (Building_1)
# Buildings 2-17 have "include": false

# Test the 1-building setup:
export CITYLEARN_SCHEMA="$PWD/data/citylearn/schema.json"
python -m scripts.test_1building

# Train with 1 building (much faster!):
python -m scripts.train_omnisafe --cfg configs/on-policy/ppo_lag_soc.yaml

# Expected changes:
# - Observation space: ~28D (shared features + 1 building)
# - Action space: 1D (single battery)
# - Episode length: 8759 steps (same)
# - Training time: ~17x faster
# - Constraint: SoC ∈ [0.1, 0.9] for Building_1 only

# To revert to 17 buildings:
# Edit data/citylearn/schema.json and set "include": true for Buildings 2-17