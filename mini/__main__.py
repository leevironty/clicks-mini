from lightning.pytorch.cli import LightningCLI

from mini.model_wrapper import ModelWrapper
from mini.dataset import TrackDataModule


def main():
    LightningCLI(
        model_class=ModelWrapper,
        datamodule_class=TrackDataModule,
    )


if __name__ == '__main__':
    main()
