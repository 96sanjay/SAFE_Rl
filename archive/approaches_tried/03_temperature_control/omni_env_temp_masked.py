"""Masked temperature CMDP wrapper for the case-study PPO policy mask.

This module keeps the existing temperature CMDP behavior intact and adds a
separate, explicitly named env variant that exposes a bounds-provider hook for
policy-side continuous masking.
"""
from __future__ import annotations

from typing import Any, ClassVar

from omnisafe.envs.core import env_register

from citylearn_safe.omni_env_temp import CityLearnTempSingleLagCMDP
from citylearn_safe.policy_action_mask_temp import (
    CityLearnTempActionBoundsProvider,
    build_policy_action_bounds_provider,
)


@env_register
class CityLearnTempMaskedSingleLagCMDP(CityLearnTempSingleLagCMDP):
    """Temperature CMDP variant with a policy-mask provider hook."""

    _support_envs: ClassVar[list[str]] = ["CityLearnTemp-Comfort-Masked-v0"]
    policy_action_bounds_provider_cls: ClassVar[type[CityLearnTempActionBoundsProvider]] = (
        CityLearnTempActionBoundsProvider
    )
    policy_action_mask_enabled: ClassVar[bool] = True

    def __init__(self, env_id: str, **kwargs: Any) -> None:
        super().__init__(env_id, **kwargs)

    def make_policy_action_bounds_provider(self) -> CityLearnTempActionBoundsProvider:
        """Factory hook used by a masked PPO adapter to obtain safe bounds."""
        return self.policy_action_bounds_provider_cls(self)

    def get_policy_action_bounds_provider(self) -> CityLearnTempActionBoundsProvider:
        """Compatibility alias for adapters that expect a getter-style hook."""
        return self.make_policy_action_bounds_provider()

    def policy_action_bounds_provider(self) -> CityLearnTempActionBoundsProvider:
        """Compatibility alias for adapters that expect a direct provider hook."""
        return self.make_policy_action_bounds_provider()

    def masked_action_bounds_provider(self) -> CityLearnTempActionBoundsProvider:
        """Compatibility alias used by older masked-action adapter code paths."""
        return self.make_policy_action_bounds_provider()


def build_temperature_policy_action_bounds_provider(
    env: Any,
) -> CityLearnTempActionBoundsProvider:
    """Build the temperature bounds provider from an env or wrapper."""
    return build_policy_action_bounds_provider(env)


__all__ = [
    "CityLearnTempMaskedSingleLagCMDP",
    "build_temperature_policy_action_bounds_provider",
]
