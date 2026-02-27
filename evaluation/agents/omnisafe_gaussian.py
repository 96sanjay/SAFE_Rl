from __future__ import annotations
from typing import Any, Dict, Optional, List
import os
import numpy as np
import torch
import torch.nn as nn
from .base import BaseAgent


class FlexibleGaussianPolicy(nn.Module):
    """Flexible Gaussian policy with variable architecture"""
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: List[int]):
        super().__init__()
        layers = []
        prev_size = obs_dim
        
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, h),
                nn.ReLU()
            ])
            prev_size = h
        
        layers.extend([
            nn.Linear(prev_size, act_dim),
            nn.Tanh()
        ])
        
        self.mean = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mean(obs)


class OmniSafeCheckpointAgent(BaseAgent):
    def __init__(self, ckpt_path: str, name: Optional[str] = None, deterministic: bool = True):
        self.ckpt_path = ckpt_path
        self.name = name or ("omnisafe_" + os.path.basename(ckpt_path).replace(".pt", ""))
        self.deterministic = deterministic
        self.model = None
        self.obs_norm = None
    
    def reset(self, env: Any) -> None:
        super().reset(env)
        
        # Load checkpoint
        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        pi_state = ckpt["pi"]
        self.obs_norm = ckpt.get("obs_normalizer", None)
        
        # Extract architecture from checkpoint keys
        # Keys look like: mean.0.weight, mean.0.bias, mean.2.weight, ...
        linear_layers = []
        
        for key in sorted(pi_state.keys()):
            if key.startswith("mean.") and key.endswith(".weight") and "log_std" not in key:
                layer_idx = int(key.split(".")[1])
                weight = pi_state[key]
                linear_layers.append((layer_idx, weight.shape))
        
        # Sort by layer index
        linear_layers.sort(key=lambda x: x[0])
        
        # Extract dimensions
        obs_dim = linear_layers[0][1][1]  # First layer input dim
        act_dim = linear_layers[-1][1][0]  # Last layer output dim
        
        # Hidden sizes are output dims of intermediate layers
        hidden_sizes = [shape[0] for idx, shape in linear_layers[:-1]]
        
        print(f"[{self.name}] Detected architecture:")
        print(f"  Observation dim: {obs_dim}")
        print(f"  Hidden sizes: {hidden_sizes}")
        print(f"  Action dim: {act_dim}")
        
        # Create model
        self.model = FlexibleGaussianPolicy(obs_dim, act_dim, hidden_sizes)
        self.model.load_state_dict(pi_state, strict=True)
        self.model.eval()
        
        print(f"[{self.name}] ✓ Checkpoint loaded!")
    
    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if not self.obs_norm:
            return obs
        
        mean = self.obs_norm["_mean"].cpu().numpy()
        var = self.obs_norm["_var"].cpu().numpy()
        eps = 1e-8
        return (obs - mean) / np.sqrt(var + eps)
    
    def act(self, obs: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(f"{self.name} not initialized. Call reset() first.")
        
        x = np.asarray(obs, dtype=np.float32).reshape(-1)
        x = self._normalize_obs(x)
        
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            mean = self.model(t).squeeze(0).cpu().numpy()
        
        return np.asarray(mean, dtype=float)
