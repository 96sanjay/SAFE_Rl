from __future__ import annotations
from typing import Any, Dict
import numpy as np

class BaseAgent:
    """Minimal interface: implement act(obs, info) -> action."""
    name: str = "base"

    def reset(self, env: Any) -> None:
        """Called once at episode start."""
        self.env = env

    def act(self, obs: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        raise NotImplementedError
