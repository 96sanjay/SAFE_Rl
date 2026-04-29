import omnisafe
import citylearn_safe.omni_env

custom_cfgs = {
    "seed": 42,
    "train_cfgs": {
        "device": "cpu",
        "parallel": 1,
        "torch_threads": 6,
        "total_steps": 875_900,
        "vector_env_nums": 1,
    },
    "algo_cfgs": {
        "steps_per_epoch": 8759,
        "batch_size": 128,
        "obs_normalize": True,
        "reward_normalize": False,
        "cost_normalize": False,
    },
    "model_cfgs": {
        "actor": {"hidden_sizes": [512, 512, 256], "activation": "relu", "lr": None},
        "critic": {"hidden_sizes": [512, 512, 256], "activation": "relu", "lr": 0.001},
    },
    "lagrange_cfgs": {
        "cost_limit": 1000.0,
        "lagrangian_multiplier_init": 0.1,
        "lambda_lr": 0.005,
        "lambda_optimizer": "Adam",
    },
    "logger_cfgs": {
        "log_dir": "./runs/trpolag_c1_shaped",
        "save_model_freq": 10,
    },
}

agent = omnisafe.Agent("TRPOLag", "CityLearnSafety-SoC-v0", custom_cfgs=custom_cfgs)
agent.learn()
