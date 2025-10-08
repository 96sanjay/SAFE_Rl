# scripts/train_omnisafe.py
from __future__ import annotations
import argparse, yaml
import omnisafe
import scripts.register_env  # ensures Simple-v0 -> your env

def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(open(cfg_path, "r"))
    algo = cfg["algo"]
    env_id = cfg["env_id"]  # should be "Simple-v0"

    allowed = (
        "train_cfgs","algo_cfgs","logger_cfgs","lagrange_cfgs",
        "model_cfgs","save_cfgs","env_cfgs","reward_model_cfgs",
    )
    custom_cfgs = {k: v for k, v in cfg.items() if k in allowed}

    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
    agent.learn()  # keep it simple for now (we can add saving later)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    args = ap.parse_args()
    main(args.cfg)
