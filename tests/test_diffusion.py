from pytest import fixture, mark, Session, skip

from audio_diffusion_pytorch import UNetV0
from mini.diffusion_inpaint import DiffusionInpaint
import torch

from lightning import Trainer, seed_everything
from mini.dataset import TrackDataModule, TracksDataset, Degradation, ActivityDetector
from mini.model_wrapper import ModelWrapper
from lightning.pytorch.loggers import TensorBoardLogger


from mini.parametrisations import EDMParams, MaskGenerator



@fixture
def one_test_only(request):
    def resolve(node = request.node):
        if type(node) is Session:
            if node.testscollected != 1:
                skip(reason='Should be the only test collected')
        else:
            resolve(node=node.parent)
    
    return resolve




@fixture
def dummy_net() -> torch.nn.Module:
    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, x: torch.Tensor, time: torch.Tensor):
            assert len(x.shape) == 3
            dim_b, dim_c, dim_t = x.shape
            assert dim_c == 2, 'The std test assumes two input channels, (mask * y + ~mask * x_i) and mask'
            mask = x[:, -1]
            x = x[:, 0]
            assert time.shape == (dim_b,)
            # return torch.ones(size=(dim_b, 1, dim_t), device=x.device)
            return torch.randn(size=(dim_b, 1, dim_t), device=x.device)

    return Dummy()



@fixture
def overfitter() -> torch.nn.Module:
    # class Model(torch.nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.stack = torch.nn.Sequential(
    #             torch.nn.Conv1d(in_channels=2, out_channels=16, kernel_size=2 ** 12 + 1, padding=2 ** 11),
    #             torch.nn.ReLU(),
    #             torch.nn.GroupNorm(4, 16),
    #             torch.nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2 ** 12 + 1, padding=2 ** 11),
    #             torch.nn.ReLU(),
    #             torch.nn.GroupNorm(4, 16),
    #             torch.nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2 ** 12 + 1, padding=2 ** 11),
    #             torch.nn.ReLU(),
    #             torch.nn.GroupNorm(4, 16),
    #             torch.nn.Conv1d(in_channels=16, out_channels=1, kernel_size=2 ** 12 + 1, padding=2 ** 11),
    #         )
    #     def forward(self, x, **_):
    #         return self.stack(x)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.out = torch.nn.Parameter(torch.randn((1, 1, 2 ** 14)))
        def forward(self, x, **_):
            return self.out
    
    return Model()




@fixture
def adp_net() -> torch.nn.Module:
    return UNetV0(
        dim=1,
        in_channels=2,
        out_channels=1,
        channels=[16, 32, 32, 64, 64, 64],
        factors=[2, 4, 4, 4, 4, 4],
        items=[2, 2, 2, 2, 2, 2],
        attentions=[0, 0, 2, 2, 2, 2],
        attention_features=256,
        attention_heads=16,
        use_time_conditioning=True,
        modulation_features=256,
    )


@fixture
def diffusion_module(
        dummy_net: torch.nn.Module,
        edm_parameters: EDMParams,
        mask_generator: MaskGenerator,
    ) -> DiffusionInpaint:
    return DiffusionInpaint(
        net=dummy_net,
        edm_params=edm_parameters,
        mask_generator=mask_generator,
    )


def test_net_d_preconditioning(diffusion_module: DiffusionInpaint):
    # torch.manual_seed(123)
    mask = diffusion_module.mask_generator(32, 44100, device=torch.device('cpu'))
    target_std = diffusion_module.edm_params.sigma_data
    x0 = torch.randn((32, 1, 44100)) * target_std
    noise = torch.randn_like(x0) * ~mask[:, None]
    n_fe = 10
    with torch.no_grad():
        for sigma in diffusion_module.edm_params.time_steps(n_fe):
            x_i = x0 + noise * sigma
            x_hat = diffusion_module.net_d(x_i=x_i, mask=mask, y=mask[:, None]*x0, sigma=sigma)
            assert x_hat.shape == (32, 1, 44100)
            assert abs(x_hat.mean()) < 0.001


def test_sampling(diffusion_module: DiffusionInpaint):
    device = torch.device('mps')
    diffusion_module.to(device)
    mask = diffusion_module.mask_generator(32, 44100, device=device)
    y = torch.rand(32, 1, 44100, device=device)
    out, steps = diffusion_module.sample(y=y, mask=mask, n_fe=20)
    assert out.shape == (32, 1, 44100)
    for step in steps:
        assert step.shape == (32, 1, 44100)


def test_train(diffusion_module: DiffusionInpaint):
    device = torch.device('mps')
    diffusion_module.to(device)
    x0 = torch.rand(32, 1, 44100, device=device)
    diffusion_module.train(x0=x0)


# @mark.skip(reason='Slow training debug run')
def test_actual_train_overfit(overfitter, adp_net, edm_parameters, mask_generator, one_test_only):
    one_test_only()
    adp_net = overfitter
    # print(type(request))
    # return 
    samples = 2 ** 14
    detector = ActivityDetector(min_power=0.01, min_samples_activity=samples, min_samples_silence=samples // 2)
    dataset = TracksDataset('~/datasets/musqdb-hq', samples=samples, activity_detector=detector)
    assert len(dataset) > 1000
    
    data_wrapper = TrackDataModule(
        dataset=dataset,
        batch_size=1,
    )

    model_wrapper = ModelWrapper(
        net=adp_net,
        edm_params=edm_parameters,
        mask_generator=mask_generator,
        lr=0.01,
        # weight_decay=0.01,
    )

    with model_wrapper.overfit_mode():
        seed_everything(152)
        trainer = Trainer(
            deterministic=True,
            overfit_batches=1,
            default_root_dir='.junk',
            max_steps=5000,
            accumulate_grad_batches=1,
            log_every_n_steps=1,
            logger=TensorBoardLogger(
                save_dir='.logs_test',
                name='tensorboard',
                log_graph=True,
                # version='custom_diffusion_bigger_v',
            ),
        )
        trainer.fit(model=model_wrapper, datamodule=data_wrapper)
