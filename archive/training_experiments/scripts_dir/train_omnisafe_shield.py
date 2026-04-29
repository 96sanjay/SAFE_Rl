#!/usr/bin/env python
"""Train with shield env. Imports all env registrations."""
from __future__ import annotations
import argparse, yaml, omnisafe
import citylearn_safe.omni_env
import citylearn_safe.omni_env_v2
import citylearn_safe.omni_env_v2_shield

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    allowed = ("train_cfgs","algo_cfgs","logger_cfgs","lagrange_cfgs","model_cfgs","save_cfgs","env_cfgs","reward_model_cfgs")
    custom = {k: v for k, v in cfg.items() if k in allowed}
    omnisafe.Agent(cfg["algo"], cfg["env_id"], custom_cfgs=custom).learn()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cfg", required=True)
    main(ap.parse_args().cfg)
