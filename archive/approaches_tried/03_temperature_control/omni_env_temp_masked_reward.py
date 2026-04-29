"""Reward-only masked temperature CMDP wrapper.

This env id exists to isolate the shielded reward-optimization experiment from
the lagrangian case-study path. The underlying dynamics, reward, and comfort
cost logging remain identical to the masked temperature CMDP; only the env id
and policy-mask defaults are separated for experiment hygiene.
"""
from __future__ import annotations

from typing import Any, ClassVar

from omnisafe.envs.core import env_register

from citylearn_safe.omni_env_temp_masked import CityLearnTempMaskedSingleLagCMDP


@env_register
class CityLearnTempMaskedRewardCMDP(CityLearnTempMaskedSingleLagCMDP):
    """Temperature CMDP variant for shielded reward-only PPO experiments."""

    _support_envs: ClassVar[list[str]] = ["CityLearnTemp-Comfort-Masked-Reward-v0"]

    def __init__(self, env_id: str, **kwargs: Any) -> None:
        super().__init__(env_id, **kwargs)


__all__ = ["CityLearnTempMaskedRewardCMDP"]
