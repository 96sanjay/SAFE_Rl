# scripts/train_omnisafe.py
from __future__ import annotations
import argparse, yaml
import gymnasium as gym
import omnisafe

# make sure our env is registered with Gym
import scripts.register_env  # noqa: F401


def _whitelist_env_id(env_id: str) -> None:
    """Monkeypatch all OmniSafe modules that gate on support_envs()."""
    modules = []
    for mod_path in [
        "omnisafe.algorithms.algo_wrapper",
        "omnisafe.adapter.online_adapter",
        "omnisafe.adapter.onpolicy_adapter",
        "omnisafe.adapter.offpolicy_adapter",
    ]:
        try:
            mod = __import__(mod_path, fromlist=["support_envs"])
            if hasattr(mod, "support_envs"):
                modules.append(mod)
        except Exception:
            pass

    for mod in modules:
        orig = mod.support_envs

        def patched_support_envs(orig=orig):
            s = orig()
            seq = list(s)
            if env_id not in seq:
                seq.append(env_id)
            return type(s)(seq) if not isinstance(s, list) else seq

        mod.support_envs = patched_support_envs


def main(cfg_path: str) -> None:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    algo = cfg["algo"]                  # e.g., "PPOLag"
    env_id = cfg["env_id"]              # "CityLearnSafety-SoC-v0"

    # only pass sections OmniSafe expects
    allowed = (
        "train_cfgs", "algo_cfgs", "logger_cfgs", "lagrange_cfgs",
        "model_cfgs", "save_cfgs", "env_cfgs", "reward_model_cfgs",
    )
    custom_cfgs = {k: v for k, v in cfg.items() if k in allowed}

    # whitelist our env id everywhere OmniSafe checks
    _whitelist_env_id(env_id)

    # create the agent using env_id STRING (OmniSafe builds its wrappers internally)
    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)

    # ---- train
    agent.learn()

    # ---- save (robust to different attribute names)
    def try_save(agent_obj, out_dir="checkpoints/ppo_lag_soc"):
        import os, torch
        os.makedirs(out_dir, exist_ok=True)
        for attr in ("actor_critic", "policy", "model"):
            obj = getattr(agent_obj, attr, None)
            if obj is not None and hasattr(obj, "state_dict"):
                torch.save(obj.state_dict(), os.path.join(out_dir, f"{attr}.pt"))
                print(f"[save] Wrote {attr}.pt to {out_dir}")
                return
        print("[save] No known policy attribute found; skipping.")

    try_save(agent)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    args = ap.parse_args()
    main(args.cfg)
