from lightning import LightningModule
import torch
import numpy
import random
from contextlib import contextmanager

from audio_diffusion_pytorch import UNetV0, VDiffusion, VSampler, VInpainter
from mini.diffusion_inpaint import DiffusionInpaint, EDMParams, MaskGenerator


class UNet(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self._net = UNetV0(**kwargs)
    
    def forward(self, *args, **kwargs):
        return self._net(*args, **kwargs)


class ModelWrapper(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        edm_params: EDMParams,
        mask_generator: MaskGenerator,
        lr: float = 0.001,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        # self.net = net
        # self.diffusion = VDiffusion(net)
        self.lr = lr
        self.weight_decay = weight_decay
        self._is_overfit_mode = False
        self.diffusion = DiffusionInpaint(
            net=net,
            edm_params=edm_params,
            mask_generator=mask_generator,
        )
    
    @contextmanager
    def overfit_mode(self):
        self._is_overfit_mode = True
        try:
            yield
        finally:
            self._is_overfit_mode = False


    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
    
    def on_train_batch_start(self, batch: torch.Any, batch_idx: int):
        if self._is_overfit_mode:
            torch.manual_seed(123)
            numpy.random.seed(321)
            random.seed(234)
        
    
    def training_step(self, batch: torch.Tensor):
        # time = torch.rand((batch.shape[0]), device=batch.device)
        batch = batch.unsqueeze(1)
        loss, logs = self.diffusion(batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss

