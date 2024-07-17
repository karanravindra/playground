import argparse

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchinfo import summary
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

from nn_zoo.datamodules import MNISTDataModule
from nn_zoo.models.components import ResidualStack, DepthwiseSeparableConv2d
from nn_zoo.trainers import AutoEncoderTrainer

class AutoEncoder(nn.Module):
    def __init__(self, width: int, depth: int=1):
        def block(in_channels, out_channels):
            return nn.Sequential(
                nn.BatchNorm2d(in_channels),
                DepthwiseSeparableConv2d(in_channels, out_channels, 3, padding=1),
                nn.GELU(),
                nn.BatchNorm2d(out_channels),
                DepthwiseSeparableConv2d(out_channels, out_channels, 3, padding=1),
                nn.GELU(),
            )
        
        def stack(in_channels, out_channels, n_blocks = depth):
            return nn.Sequential(
                *[
                    block(in_channels, out_channels)
                    if i == 0
                    else block(out_channels, out_channels)
                    for i in range(n_blocks)
                  ]
                )

        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            stack(1, width),
            nn.MaxPool2d(2),
            stack(width, width*2),
            # nn.MaxPool2d(2),
            # stack(width*2, width*4),
            # nn.MaxPool2d(2),
            # stack(width*4, width*8),
            
        )
        self.decoder = nn.Sequential(
            # stack(width*8, width*4),
            # nn.Upsample(scale_factor=2),
            # stack(width*4, width*2),
            # nn.Upsample(scale_factor=2),
            stack(width*2, width),
            nn.Upsample(scale_factor=2),
            stack(width, 1),
            nn.Sigmoid()
        )
        
        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)
        
    def init_weights(self, m):
        if type(m) == nn.Conv2d:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
    
    @clas
    def loss(self, x, y):
        return F.mse_loss(x, y)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a MNIST autoencoder")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=4,
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
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=25, help="Number of training epochs"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
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
    dm = MNISTDataModule(
        data_dir="data",
        dataset_params={
            "download": True,
            "transform": torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize((32, 32)),
                    torchvision.transforms.ToTensor(),
                ]
            ),
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
            depth=args.depth
        ),
        dm=dm,
        optim=args.optim,
        optim_kwargs={"lr": args.learning_rate},
    )

    summary(classifier_trainer.model, input_size=(args.batch_size, 1, 32, 32), depth=2)

    logger = WandbLogger(project="mnist-autoencoder", name="autoencoder", log_model=True)
    logger.watch(classifier_trainer.model, log="all")
    logger.log_hyperparams(vars(args))

    trainer = Trainer(max_epochs=args.epochs, logger=logger, default_root_dir="logs", check_val_every_n_epoch=1)
    trainer.fit(classifier_trainer, dm)
    trainer.test(classifier_trainer, dm)


if __name__ == "__main__":
    args = parse_args()
    main(args)
