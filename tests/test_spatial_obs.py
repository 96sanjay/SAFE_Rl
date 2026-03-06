#!/usr/bin/env python3
"""Test that spatial observations are appended when CITYLEARN_SPATIAL_OBS=1.

Uses the 5-building schema by default, but works with any schema.
"""
import os, sys
import numpy as np

os.chdir("/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork")
sys.path.insert(0, os.getcwd())

# Default to 5-building schema; override with CITYLEARN_SCHEMA env var
if "CITYLEARN_SCHEMA" not in os.environ:
    os.environ["CITYLEARN_SCHEMA"] = os.path.join(
        os.getcwd(), "data/citylearn_challenge_2022_phase_all_plus_evs/schema.json"
    )
os.environ["CITYLEARN_STEMS_P_BUILDING_MAX"] = "4.6083"
os.environ["CITYLEARN_STEMS_P_GRID_MAX"] = "29.6915"
os.environ["CITYLEARN_REWARD_TYPE"] = "stems"
os.environ["CITYLEARN_EV_MISSING_ACTION_MODE"] = "assume_zero"

# Test WITHOUT spatial obs
os.environ["CITYLEARN_SPATIAL_OBS"] = "0"

from citylearn.citylearn import CityLearnEnv
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper

base = CityLearnEnv(schema=os.environ["CITYLEARN_SCHEMA"], central_agent=True)
num_buildings = len(base.buildings)
print(f"Schema: {os.environ['CITYLEARN_SCHEMA']}")
print(f"Num buildings: {num_buildings}")

safety = CityLearnSafetyEnvV3(base)
forecast = ForecastObsWrapper(safety, forecast_horizon=24)
obs_no_spatial, _ = forecast.reset()
dim_no_spatial = len(np.asarray(obs_no_spatial).ravel())

# Test WITH spatial obs
os.environ["CITYLEARN_SPATIAL_OBS"] = "1"
from citylearn_safe.stems_obs_wrapper import SpatialGraphFeaturesWrapper

base2 = CityLearnEnv(schema=os.environ["CITYLEARN_SCHEMA"], central_agent=True)
safety2 = CityLearnSafetyEnvV3(base2)
forecast2 = ForecastObsWrapper(safety2, forecast_horizon=24)
spatial = SpatialGraphFeaturesWrapper(forecast2, num_buildings=num_buildings, p_building_max=4.6083)
obs_with_spatial, _ = spatial.reset()
dim_with_spatial = len(np.asarray(obs_with_spatial).ravel())

expected_extra = num_buildings * 4
print(f"Without spatial: {dim_no_spatial} dims")
print(f"With spatial:    {dim_with_spatial} dims")
print(f"Difference:      {dim_with_spatial - dim_no_spatial} dims (expected {expected_extra})")

assert dim_with_spatial == dim_no_spatial + expected_extra, (
    f"Expected +{expected_extra} spatial dims, got +{dim_with_spatial - dim_no_spatial}"
)

# Check spatial features are not all zeros (headroom should be non-zero)
spatial_features = np.asarray(obs_with_spatial).ravel()[-expected_extra:]
assert not np.all(spatial_features == 0.0), "Spatial features should not be all zeros"

print(f"PASS: Spatial obs wrapper adds {expected_extra} non-zero features ({num_buildings} buildings)")
