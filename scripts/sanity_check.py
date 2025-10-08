import sys
print("python:", sys.version)

# Core libs
import numpy as np, pandas as pd
print("numpy:", np.__version__, "pandas:", pd.__version__)

# RL stack
import gymnasium as gym
print("gymnasium:", gym.__version__)

# DL stack
import torch
print("torch:", torch.__version__, "MPS available:", torch.backends.mps.is_available())

# OmniSafe
import omnisafe
print("omnisafe imported OK")

# CityLearn
import citylearn
print("citylearn imported OK")

print("✅ environment looks good.")
