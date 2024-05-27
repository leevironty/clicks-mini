# from typing import TypeVar

from einops import repeat
import torch
from torch import Tensor, log, sqrt, exp


from dataclasses import dataclass
# from math import log, sqrt
# from torch import log, sqrt
# from numpy import log, sqrt



# F = TypeVar('F', float, torch.Tensor)

@dataclass
class EDMParams:
    sigma_min: float
    sigma_max: float
    sigma_data: float
    rho: float
    p_mean: float
    p_std: float

    # @staticmethod
    # def _expand(t: Tensor) -> Tensor:
    #     return repeat(t, 'b -> b 1 1')

    def c_in(self, sigma: Tensor):
        return 1 / sqrt(sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma: Tensor):
        return sigma * self.sigma_data / sqrt(self.sigma_data ** 2 + sigma ** 2)

    def c_skip(self, sigma: Tensor):
        return self.sigma_data ** 2/ sqrt(self.sigma_data ** 2 + sigma ** 2)

    def c_noise(self, sigma: Tensor):
        return log(sigma) / 4

    def loss_weight(self, sigma: Tensor):
        return (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

    def time_steps(self, n_fe: int) -> list[float]:
        upper = self.sigma_max ** (1 / self.rho)
        lower = self.sigma_min ** (1 / self.rho)
        return [
            (upper + i / (n_fe - 1) * (lower - upper)) ** self.rho
            for i in range(n_fe)
        ] + [0]

    def normal_samples_to_time(self, samples: Tensor) -> Tensor:
        return exp(samples * self.p_std + self.p_mean).clip(self.sigma_min, self.sigma_max)
    
    def debug_time_distribution(self, linspace_params = (-2, 3, 50)):
        import matplotlib.pyplot as plt
        import numpy as np
        ts = self.normal_samples_to_time(torch.randn(100000))
        plt.hist(ts, bins=10**np.linspace(*linspace_params))
        plt.xscale('log'); plt.savefig('plot-log.png')
        plt.clf()


@dataclass
class StochasticParams:
    tmin: float
    tmax: float
    churn: float
    noise: float

    def gammas(self, time_steps: list[float]) -> list[float]:
        n_fe = len(time_steps) - 1
        # TODO: check if correct or that makes sense
        gamma = min(self.churn / n_fe, 2 ** 0.5 - 1)
        def value(t):
            if self.tmin <= t and t <= self.tmax:
                return gamma
            return 0
        return [
            value(t) for t in time_steps[:-1]
        ]


@dataclass
class MaskGenerator:
    min_len: int
    avg_len: int
    max_len: int
    min_count: int
    max_count: int
    target_share: float = 0.05  # TODO: better parametrization

    def __call__(self, dim_b: int, dim_t: int, device: torch.device):
        mask = torch.zeros((dim_b, dim_t), dtype=torch.bool, device=device)
        dist = torch.distributions.Exponential(rate=1/(self.avg_len - self.min_len))
        n_loss_events = torch.randint(
            low=self.min_count,
            high=self.max_count+1,
            size=(dim_b,)
        )
        for batch, n_events in enumerate(n_loss_events):
            n_events = n_events.item()
            assert type(n_events) is int
            durations = dist.rsample(sample_shape=torch.Size((n_events,)))
            durations = durations.fmod(self.max_len - self.min_len) + self.min_len
            durations = durations * (self.target_share * dim_t) / durations.sum()
            durations = durations.to(torch.int32)

            starts = torch.randint(0, dim_t, (n_events,))
            ends = starts + durations
            for start, end in zip(starts, ends):
                mask[batch, start:end] = True
        return ~mask
