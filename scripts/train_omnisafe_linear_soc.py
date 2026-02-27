# scripts/train_omnisafe_linear_soc.py
"""Training script that uses LINEAR SoC cost environment"""
from __future__ import annotations
import argparse, yaml
import omnisafe

import citylearn_safe.omni_env_linear_soc  # <-- Import LINEAR version

def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(open(cfg_path, "r"))
    algo = cfg["algo"]
    env_id = cfg["env_id"]
    
    allowed = (
        "train_cfgs","algo_cfgs","logger_cfgs","lagrange_cfgs",
        "model_cfgs","save_cfgs","env_cfgs","reward_model_cfgs",
    )
    custom_cfgs = {k: v for k, v in cfg.items() if k in allowed}
    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
    agent.learn()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    args = ap.parse_args()
    main(args.cfg)
