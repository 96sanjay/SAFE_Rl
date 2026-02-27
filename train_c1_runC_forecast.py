"""
Run C: C1-only TRPOLag — NO shaping, WITH ForecastObsWrapper (281-dim obs)
Tests whether better information (EV urgency, price/load/solar forecasts)
helps the agent learn C1 satisfaction without reward engineering.
"""
import omnisafe
import citylearn_safe.omni_env_forecast  # registers CityLearnSafety-Forecast-v0

custom_cfgs = {
    "seed": 42,
    "train_cfgs": {
        "device": "cpu",
        "parallel": 1,
        "torch_threads": 4,
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
        "log_dir": "./runs/trpolag_c1_forecast",
        "save_model_freq": 10,
    },
}

agent = omnisafe.Agent("TRPOLag", "CityLearnSafety-Forecast-v0", custom_cfgs=custom_cfgs)
agent.learn()
