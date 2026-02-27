"""
TRPOLag, NO PSF, focused on C1 (EV departure violations).

Hyperparameter reasoning:
  - zero-action EpCost    = 3,410  (C1=2320, C4=1023, C3=67)
  - discharge exploit     = 10,569 (baseline TRPOLag, lambda=0)
  - cost_limit = 8,000    → above zero-action so agent not overwhelmed
                          → below discharge exploit so lambda stays ACTIVE
  - lambda_lr = 0.005     → slow update, prevents overshoot+collapse to 0
                          → baseline used 0.035 which caused collapse at ep26
  - lambda_init = 0.1     → nonzero start, lambda never fully dies
  - EV shaping alpha=3.0  → dense per-step C1 signal, ~2x STEMS per step
                          → not strong enough to dominate, but guides gradient
"""
import os
import omnisafe
import citylearn_safe.omni_env  # noqa: F401  registers CityLearnSafety-SoC-v0

custom_cfgs = {
    "seed": 42,
    "train_cfgs": {
        "device": "cpu",
        "parallel": 1,
        "torch_threads": 12,
        "total_steps": 2_000_000,
        "vector_env_nums": 1,
    },
    "algo_cfgs": {
        "steps_per_epoch": 8759,   # exactly 1 episode per epoch, clean comparison
        "batch_size": 128,
        "obs_normalize": True,
        "reward_normalize": False,
        "cost_normalize": False,
    },
    "model_cfgs": {
        "actor": {
            "hidden_sizes": [512, 512, 256],
            "activation": "relu",
            "lr": None,
        },
        "critic": {
            "hidden_sizes": [512, 512, 256],
            "activation": "relu",
            "lr": 0.001,
        },
    },
    "lagrange_cfgs": {
        "cost_limit": 8000.0,
        "lagrangian_multiplier_init": 0.1,
        "lambda_lr": 0.005,
        "lambda_optimizer": "Adam",
    },
    "logger_cfgs": {
        "log_dir": "./runs/trpolag_c1_focus",
        "save_model_freq": 10,
    },
}

print("=" * 65)
print("TRPOLag C1-Focus (NO PSF)")
print("=" * 65)
print(f"  EV shaping alpha  : {os.environ.get('CITYLEARN_EV_SHAPING_ALPHA', 'NOT SET')}")
print(f"  cost_limit        : {custom_cfgs['lagrange_cfgs']['cost_limit']}")
print(f"  lambda_init       : {custom_cfgs['lagrange_cfgs']['lagrangian_multiplier_init']}")
print(f"  lambda_lr         : {custom_cfgs['lagrange_cfgs']['lambda_lr']}")
print(f"  zero-action cost  : ~3,410")
print(f"  discharge exploit : ~10,569  (lambda must stay active above this)")
print(f"  steps/epoch       : {custom_cfgs['algo_cfgs']['steps_per_epoch']} (1 full episode)")
print(f"  total epochs      : {custom_cfgs['train_cfgs']['total_steps'] // custom_cfgs['algo_cfgs']['steps_per_epoch']}")
print("=" * 65)

agent = omnisafe.Agent("TRPOLag", "CityLearnSafety-SoC-v0", custom_cfgs=custom_cfgs)
agent.learn()
