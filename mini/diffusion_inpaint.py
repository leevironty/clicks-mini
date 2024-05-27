from typing import TypeVar
import torch
from einops import repeat

from mini.parametrisations import EDMParams, StochasticParams, MaskGenerator


T = TypeVar('T')

def default(a: T | None, b: T) -> T:
    if a is None:
        return b
    return a


class DiffusionInpaint(torch.nn.Module):
    def __init__(
        self,
        net: torch.nn.Module,
        edm_params: EDMParams,
        mask_generator: MaskGenerator
    ) -> None:
        super().__init__()
        self.edm_params = edm_params
        self.net_f = net
        self.mask_generator = mask_generator


    def net_d(
        self,
        x_i: torch.Tensor,
        mask: torch.Tensor,
        y: torch.Tensor,
        sigma: torch.Tensor | float,
    ) -> torch.Tensor:
        # check shapes
        assert len(x_i.shape) == 3
        assert x_i.shape == y.shape
        dim_b, dim_c, dim_t = y.shape
        assert mask.shape == (dim_b, dim_t)
        assert dim_c == 1  # NOTE: for sanity checking a simple case
        if not isinstance(sigma, torch.Tensor):
            sigma_tensor = torch.Tensor([sigma]).to(device=y.device)
            sigma_tensor = repeat(sigma_tensor, '1 -> b', b=y.shape[0])
        else:
            sigma_tensor = sigma
        assert sigma_tensor.shape == (dim_b,)
        sigma_expanded = repeat(sigma_tensor, 'b -> b 1 1')
        c_in = self.edm_params.c_in(sigma_expanded)
        c_out = self.edm_params.c_out(sigma_expanded)
        c_skip = self.edm_params.c_skip(sigma_expanded)
        c_noise = self.edm_params.c_noise(sigma_tensor)

        mask = repeat(mask, 'b t -> b 1 t')
        # TODO: experiment with passing y separately w/o c_in scaling
        z = mask * y + ~mask * x_i
        z = c_in * z
        z = torch.concat([z, mask], dim=1)
        # TODO: could the full output be useful outside of this function?
        xhat_full = c_skip * x_i + c_out * self.net_f(z, time=c_noise)
        return mask * y + ~mask * xhat_full


    def train(self, x0: torch.Tensor):
        assert len(x0.shape) == 3
        dim_b, _, dim_t = x0.shape
        # TODO: allow alternative sampling distributions
        # time = torch.rand(dim_b, device=x0.device)
        time = torch.randn(dim_b, device=x0.device)
        time = self.edm_params.normal_samples_to_time(time)
        time_expanded = repeat(time, 'b -> b 1 1')
        mask = self.mask_generator(dim_b=dim_b, dim_t=dim_t, device=x0.device)
        mask_expanded = repeat(mask, 'b t -> b 1 t')
        noise = torch.randn_like(x0) * time_expanded
        x_i = x0 + ~mask_expanded * noise
        pred = self.net_d(x_i=x_i, mask=mask, y=mask_expanded*x0, sigma=time)
        loss_weight = self.edm_params.loss_weight(time_expanded)
        loss = torch.nn.functional.mse_loss(~mask_expanded * pred, ~mask_expanded * x0, reduce=False)
        loss = loss * loss_weight  # TODO: experiment with & without loss weight
        loss = loss.mean() / (~mask).float().mean()
        return loss, {'x0': x0, 'x0_hat': pred, 'mask': mask, 'noise': noise, 'time': time}
    
    def forward(self, x0: torch.Tensor):
        return self.train(x0)
    
    def sample(
        self,
        y: torch.Tensor,
        mask: torch.Tensor,
        n_fe: int,
        stoch_params: StochasticParams | None = None,
        use_heun: bool = True,
    ):
        with torch.no_grad():
            # check shapes
            assert len(y.shape) == 3
            assert len(mask.shape) == 2
            dim_b, _, dim_t = y.shape
            assert mask.shape == (dim_b, dim_t)

            if stoch_params is None:
                stoch_params = StochasticParams(0, 0, 0, 1)
            time_steps = self.edm_params.time_steps(n_fe)
            gammas = stoch_params.gammas(time_steps)

            x = torch.randn_like(y) * time_steps[0]

            denoising_steps: list[torch.Tensor] = []
            for i in range(n_fe):
                denoising_steps.append(x)
                eps = torch.randn_like(y) * stoch_params.noise
                t = time_steps[i]
                t_hat = (1 + gammas[i]) * t
                x_hat: torch.Tensor = x + (t_hat ** 2 - t ** 2) ** 0.5 * eps
                d = (x - self.net_d(x_i=x_hat, mask=mask, y=y, sigma=t_hat)) / t_hat
                x_prime = x_hat + (time_steps[i + 1] - t_hat) * d
                if time_steps[i + 1] != 0 and use_heun:
                    t_next = time_steps[i + 1]
                    d_prime = (x_prime - self.net_d(x_i=x_prime, mask=mask, y=y, sigma=t_next)) / t_next
                    x = x_hat + (t_next - t_hat) * (0.5 * d + 0.5 * d_prime)
                else:
                    x = x_prime
            return x, denoising_steps
