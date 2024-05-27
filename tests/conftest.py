from mini.parametrisations import EDMParams, StochasticParams, MaskGenerator


from pytest import fixture


def _edm_parameters():
    return EDMParams(
        sigma_min=0.002,
        sigma_max=80,
        sigma_data=0.5,
        rho=7,
        p_mean=-1.2,
        p_std=2,
    )


def _stoch_parameters():
    return StochasticParams(0.01, 1, 30, 1.007)


def _mask_generator():
    return MaskGenerator(
        min_len=4,
        avg_len=250,
        max_len=2000,
        min_count=8,
        max_count=16,
        target_share=0.025,
    )


@fixture
def edm_parameters():
    return _edm_parameters()


@fixture
def stoch_parameters():
    return _stoch_parameters()


@fixture
def mask_generator():
    return _mask_generator()