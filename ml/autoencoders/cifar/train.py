import argparse

from torch import nn
from torch.nn import functional as F
import torchvision
from torchinfo import summary
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

from nn_zoo.datamodules import CIFARDataModule
from nn_zoo.models.components import ResidualStack, DepthwiseSeparableConv2d
from nn_zoo.trainers import AutoEncoderTrainer


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int):
        super(Block, self).__init__()
        self.layers = nn.ModuleList([
            self._block(in_channels, out_channels)
            if i == 0
            else self._block(out_channels, out_channels)
            for i in range(num_layers)
        ])

    def _block(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            DepthwiseSeparableConv2d(in_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.layers[0](x)
        for layer in self.layers[1:]:
            x = x + layer(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, width: int, depth: int):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            Block(1, width, depth),
            nn.MaxPool2d(2),
            Block(width, width * 2, depth),
            nn.MaxPool2d(2),
            Block(width * 2, width * 4, depth),
            nn.MaxPool2d(2),
            Block(width * 4, width * 4, depth),
        )
        self.decoder = nn.Sequential(
            Block(width * 4, width * 4, depth),
            nn.Upsample(scale_factor=2),
            Block(width * 4, width * 2, depth),
            nn.Upsample(scale_factor=2),
            Block(width * 2, width, depth),
            nn.Upsample(scale_factor=2),
            Block(width, 1, depth),
            nn.Sigmoid(),
        )

    def forward(self, x, y=None):
        if y is not None:
            # OVERRIDE `y`
            y = x

        x = self.encoder(x)
        x = self.decoder(x)

        if y is None:
            return x
        else:
            loss = F.binary_cross_entropy(x, y)
            return x, loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CIFAR autoencoder")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=8,
        help="Width of the first layer of the model",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Depth of the convolutional stack",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adam",
        help="Optimizer to use for training",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for the dataloader",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=4,
        help="Prefetch factor for the dataloader",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        default=True,
        help="Pin memory for the dataloader",
    )
    parser.add_argument(
        "--persistent_workers",
        action="store_true",
        default=True,
        help="Use persistent workers for the dataloader",
    )

    return parser.parse_args()


def main(args):
    dm = CIFARDataModule(
        data_dir="data",
        dataset_params={
            "download": True,
            "transform": torchvision.transforms.Compose([
                torchvision.transforms.Resize((32, 32)),
                torchvision.transforms.ToTensor(),
            ]),
        },
        loader_params={
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "persistent_workers": args.persistent_workers,
            "pin_memory": args.pin_memory,
            "prefetch_factor": args.prefetch_factor,
        },
    )

    classifier_trainer = AutoEncoderTrainer(
        model=AutoEncoder(
            width=args.width,
            depth=args.depth,
        ),
        dm=dm,
        optim=args.optim,
        optim_kwargs={"lr": args.learning_rate},
    )

    summary(classifier_trainer.model, input_size=(args.batch_size, 3, 32, 32), depth=2)

    logger = WandbLogger(
        project="cifar-autoencoder", name="autoencoder", log_model=True
    )
    logger.watch(classifier_trainer.model, log="all")
    logger.log_hyperparams(vars(args))

    trainer = Trainer(
        max_epochs=args.epochs,
        logger=logger,
        default_root_dir="logs",
        accumulate_grad_batches=1,
        check_val_every_n_epoch=8,
    )
    trainer.fit(classifier_trainer)
    trainer.test(classifier_trainer)


if __name__ == "__main__":
    args = parse_args()
    main(args)
