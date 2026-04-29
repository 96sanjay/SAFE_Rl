import omnisafe, citylearn_safe.omni_env
agent = omnisafe.Agent("TRPOLag", "CityLearnSafety-SoC-v0", custom_cfgs={
    "seed": 42,
    "train_cfgs": {"device": "cpu", "parallel": 1, "torch_threads": 4, "total_steps": 875900, "vector_env_nums": 1},
    "algo_cfgs": {"steps_per_epoch": 8759, "batch_size": 128, "obs_normalize": True, "reward_normalize": False, "cost_normalize": False},
    "model_cfgs": {
        "actor": {"hidden_sizes": [512, 512, 256], "activation": "relu", "lr": None},
        "critic": {"hidden_sizes": [512, 512, 256], "activation": "relu", "lr": 0.001},
    },
    "lagrange_cfgs": {"cost_limit": 1000.0, "lagrangian_multiplier_init": 10.0, "lambda_lr": 0.05, "lambda_optimizer": "Adam"},
    "logger_cfgs": {"log_dir": "./runs/c1_A_pure", "save_model_freq": 10},
})
agent.learn()
