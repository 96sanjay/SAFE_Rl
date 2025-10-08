# scripts/benchmark_algos.py
import os, yaml, subprocess, sys

ALGOS = [
    "PPO","PPOLag","TRPO","TRPOLag","RCPO",
    "SAC","SACLag","TD3","TD3Lag","DDPG","DDPGLag",
]

BASE = {
  "env_id": "CityLearnSafety-SoC-v0",
  "train_cfgs": {"total_steps": 87590, "vector_env_nums": 1},
  "algo_cfgs": {"steps_per_epoch": 8759, "obs_normalize": True, "reward_normalize": True},
  "logger_cfgs": {"use_tensorboard": True, "use_wandb": False},
}

def cfg_for(algo):
    cfg = {**BASE, "algo": algo}
    # on-policy add target_kl
    if any(algo.lower().startswith(p) for p in ["ppo","trpo","rcpo","cpo","pcpo"]):
        cfg.setdefault("algo_cfgs", {}).update({"target_kl": 0.03})
    return cfg

def main():
    os.makedirs("configs", exist_ok=True)
    for a in ALGOS:
        path = f"configs/_auto_{a.lower()}_soc.yaml"
        yaml.safe_dump(cfg_for(a), open(path, "w"))
        print(f"\n=== Running {a} ===")
        cmd = [sys.executable, "-m", "scripts.train_omnisafe", "--cfg", path]
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
