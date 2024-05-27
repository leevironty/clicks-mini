from pytest import approx

from mini.parametrisations import EDMParams, StochasticParams, MaskGenerator
import torch



def test_types(edm_parameters: EDMParams):
    sigma = torch.Tensor([0.2, 0.5])
    c_in = edm_parameters.c_in(sigma)
    assert type(c_in) is torch.Tensor
    assert type(sigma) is torch.Tensor


def test_schedule(edm_parameters: EDMParams):
    n_fe = 20
    steps = edm_parameters.time_steps(n_fe)

    assert len(steps) == n_fe + 1
    # correct start and end points
    assert edm_parameters.sigma_max == approx(steps[0])
    assert edm_parameters.sigma_min == approx(steps[-2])
    assert 0 == steps[-1]
    # decreases monotonically
    for x, y in zip(steps[:-1], steps[1:]):
        assert x > y



def test_params(edm_parameters: EDMParams):
    n_fe = 20
    stoch_params = StochasticParams(0.01, 1, 30, 1.007)
    times = edm_parameters.time_steps(n_fe)
    gammas = stoch_params.gammas(times)
    assert len(gammas) == n_fe
    lower = 0
    upper = 2 ** 0.5 - 1
    for g in gammas:
        assert lower <= g and (g == approx(upper) or g < upper)


def test_mask_generator(mask_generator):
    dim_b = 32
    dim_t = 44100
    mask = mask_generator(dim_b=dim_b, dim_t=dim_t, device=torch.device('cpu'))
    assert mask.shape == (dim_b, dim_t)
    shares = (~mask).float().mean(dim=-1)

    # total share of mask is about constant
    assert all(abs(shares - mask_generator.target_share) / mask_generator.target_share - 1 < 0.01)
    mask_starts = (mask.float().diff() > 0).sum(dim=-1)
    # valid number of blocks
    # assert all(min_count <= mask_starts)  # flaky test: masks can be combined
    assert all(mask_starts <= mask_generator.max_count)