import argparse

from torch import nn
from torch.nn import functional as F
import torchvision
from torchinfo import summary
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

from nn_zoo.datamodules import CIFARDataModule
from nn_zoo.models.components import ConvStack
from nn_zoo.trainers import ClassifierTrainer


class Classifer(nn.Module):
    def __init__(self, width: int, depth: int, dropout_p: float, use_linear_norm: bool):
        super(Classifer, self).__init__()

        def down_block(in_channels, out_channels, depth):
            return nn.Sequential(
                ConvStack(in_channels, out_channels, depth),
                nn.MaxPool2d(2),
            )

        self.backbone = nn.Sequential(
            down_block(3, width, depth),
            down_block(width, width * 2, depth),
            down_block(width * 2, width * 4, depth),
            down_block(width * 4, width * 8, depth),
            ConvStack(width * 8, depth * 8, depth),
        )
        norm = nn.LazyBatchNorm1d() if use_linear_norm else nn.Identity()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.GELU(),
            nn.Dropout(dropout_p),
            norm,
            nn.LazyLinear(10),
        )

    def forward(self, x, y=None):
        x = self.backbone(x)
        x = self.classifier(x)

        if y is None:
            return x
        else:
            loss = F.cross_entropy(x, y)
            return x, loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train a MNIST classifier")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
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
        "--dropout_prob",
        type=float,
        default=0.1,
        help="Dropout probability for the classifier",
    )
    parser.add_argument(
        "--use_linear_norm",
        action="store_true",
        help="Use a linear layer for normalization",
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
        "--epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of workers for the dataloader",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="Prefetch factor for the dataloader",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Pin memory for the dataloader",
    )
    parser.add_argument(
        "--persistent_workers",
        action="store_true",
        help="Use persistent workers for the dataloader",
    )

    return parser.parse_args()


def main(args):
    dm = CIFARDataModule(
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

    classifier_trainer = ClassifierTrainer(
        model=Classifer(
            width=args.width,
            depth=args.depth,
            dropout_p=args.dropout_prob,
            use_linear_norm=args.use_linear_norm,
        ),
        dm=dm,
        optim=args.optim,
        optim_kwargs={"lr": args.learning_rate},
    )

    summary(classifier_trainer.model, input_size=(args.batch_size, 1, 32, 32))

    logger = WandbLogger(project="cifar-classifier", name="classifier", log_model=True)
    logger.watch(classifier_trainer.model, log="all")
    logger.log_hyperparams(vars(args))

    trainer = Trainer(max_epochs=args.epochs, logger=logger, default_root_dir="logs")
    trainer.fit(classifier_trainer)
    trainer.test(classifier_trainer)


if __name__ == "__main__":
    args = parse_args()
    main(args)
