"""Custom on-policy buffers that carry policy-side action mask bounds."""
from __future__ import annotations

import torch

from omnisafe.common.buffer.onpolicy_buffer import OnPolicyBuffer
from omnisafe.common.buffer.vector_onpolicy_buffer import VectorOnPolicyBuffer
from omnisafe.utils import distributed


class MaskedOnPolicyBuffer(OnPolicyBuffer):
    """On-policy buffer with safe_min/safe_max fields for masked PPO updates."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        act_shape = self.data['act'].shape[1:]
        self.add_field('safe_min', act_shape, torch.float32)
        self.add_field('safe_max', act_shape, torch.float32)

    def get(self) -> dict[str, torch.Tensor]:
        self.ptr, self.path_start_idx = 0, 0
        data = {
            'obs': self.data['obs'],
            'act': self.data['act'],
            'target_value_r': self.data['target_value_r'],
            'adv_r': self.data['adv_r'],
            'logp': self.data['logp'],
            'discounted_ret': self.data['discounted_ret'],
            'adv_c': self.data['adv_c'],
            'target_value_c': self.data['target_value_c'],
            'safe_min': self.data['safe_min'],
            'safe_max': self.data['safe_max'],
        }
        adv_mean, adv_std, *_ = distributed.dist_statistics_scalar(data['adv_r'])
        cadv_mean, *_ = distributed.dist_statistics_scalar(data['adv_c'])
        if self._standardized_adv_r:
            data['adv_r'] = (data['adv_r'] - adv_mean) / (adv_std + 1e-8)
        if self._standardized_adv_c:
            data['adv_c'] = data['adv_c'] - cadv_mean
        return data


class MaskedVectorOnPolicyBuffer(VectorOnPolicyBuffer):
    """Vectorized buffer variant with safe_min/safe_max fields."""

    def __init__(  # pylint: disable=super-init-not-called,too-many-arguments
        self,
        obs_space,
        act_space,
        size,
        gamma,
        lam,
        lam_c,
        advantage_estimator,
        penalty_coefficient,
        standardized_adv_r,
        standardized_adv_c,
        num_envs=1,
        device=torch.device('cpu'),
    ) -> None:
        self._num_buffers = num_envs
        self._standardized_adv_r = standardized_adv_r
        self._standardized_adv_c = standardized_adv_c
        self.buffers = [
            MaskedOnPolicyBuffer(
                obs_space=obs_space,
                act_space=act_space,
                size=size,
                gamma=gamma,
                lam=lam,
                lam_c=lam_c,
                advantage_estimator=advantage_estimator,
                penalty_coefficient=penalty_coefficient,
                device=device,
            )
            for _ in range(num_envs)
        ]
