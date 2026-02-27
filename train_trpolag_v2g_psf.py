import yaml, omnisafe
import citylearn_safe.omni_env_v2g_psf

with open("configs/on-policy/trpolag_v2g_psf.yaml") as f:
    raw = yaml.safe_load(f)

custom_cfgs = raw["defaults"]
seed = custom_cfgs.pop("seed", 42)
custom_cfgs["seed"] = seed

print(f"[Train] TRPOLag on CityLearnV2GPSF-v0, seed={seed}")
agent = omnisafe.Agent("TRPOLag", "CityLearnV2GPSF-v0", custom_cfgs=custom_cfgs)
agent.learn()
print("[Train] Done.")
