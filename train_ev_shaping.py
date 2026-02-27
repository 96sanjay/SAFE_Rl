"""
Training script for EV reward shaping experiment.
Uses TRPOLag with PSF V2G + 3 key changes:
  1. EV reward shaping (CITYLEARN_EV_SHAPING_ALPHA=5.0)
  2. w_track=10 (PSF_W_TRACK=10.0)  
  3. cost_limit=5.0 (in config below)
"""
import os
import omnisafe

# Register the env
import citylearn_safe.omni_env_v2g_psf  # noqa: F401

SEED = int(os.environ.get("TRAIN_SEED", "42"))
TOTAL_STEPS = int(os.environ.get("TRAIN_TOTAL_STEPS", "8000000"))
STEPS_PER_EPOCH = int(os.environ.get("TRAIN_STEPS_PER_EPOCH", "17518"))  # 2 episodes

custom_cfgs = {
    "seed": SEED,
    "train_cfgs": {
        "device": "cpu",
        "parallel": 1,
        "torch_threads": 12,
        "total_steps": TOTAL_STEPS,
        "vector_env_nums": 1,
    },
    "algo_cfgs": {
        "steps_per_epoch": STEPS_PER_EPOCH,
        "batch_size": 256,
        "obs_normalize": True,
        "reward_normalize": False,
        "cost_normalize": False,
        "update_iters": 40,
        "use_max_grad_norm": True,
        "max_grad_norm": 0.5,
    },
    "model_cfgs": {
        "actor": {
            "hidden_sizes": [512, 512, 256],
            "activation": "relu",
            "lr": 3e-4,
        },
        "critic": {
            "hidden_sizes": [512, 512, 256],
            "activation": "relu",
            "lr": 1e-3,
        },
    },
    "lagrange_cfgs": {
        "cost_limit": 5.0,          # KEY CHANGE: was 25.0
        "lagrangian_multiplier_init": 0.01,  # Start nonzero so lambda activates faster
        "lambda_lr": 0.05,          # Slightly faster lambda learning
        "lambda_optimizer": "Adam",
    },
    "logger_cfgs": {
        "log_dir": "./runs/ev_shaping",
        "save_model_freq": 5,
    },
}

print("=" * 70)
print("EV SHAPING EXPERIMENT")
print("=" * 70)
print(f"  EV shaping alpha:  {os.environ.get('CITYLEARN_EV_SHAPING_ALPHA', 'NOT SET')}")
print(f"  PSF w_track:       {os.environ.get('PSF_W_TRACK', 'NOT SET')}")
print(f"  Reward type:       {os.environ.get('CITYLEARN_REWARD_TYPE', 'NOT SET')}")
print(f"  Cost limit:        {custom_cfgs['lagrange_cfgs']['cost_limit']}")
print(f"  Lambda init:       {custom_cfgs['lagrange_cfgs']['lagrangian_multiplier_init']}")
print(f"  Seed:              {SEED}")
print(f"  Total steps:       {TOTAL_STEPS}")
print("=" * 70)

agent = omnisafe.Agent("TRPOLag", "CityLearnV2GPSF-v0", custom_cfgs=custom_cfgs)
agent.learn()
