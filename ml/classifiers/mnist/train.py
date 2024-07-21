import argparse

from torch import nn
from torch.nn import functional as F
import torchvision
from torchinfo import summary
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

from nn_zoo.datamodules import MNISTDataModule
from nn_zoo.trainers import ClassifierTrainer


class Classifer(nn.Module):
    def __init__(self, width: int, depth: int):
        super(Classifer, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(10),
        )

    def forward(self, x, *args, **kwargs):
        x = self.backbone(x)
        x = self.classifier(x)

        return x

    @classmethod
    def loss(cls, y_hat, y):
        return F.cross_entropy(y_hat, y)


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
        default=2,
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

    classifier_trainer = ClassifierTrainer(
        model=Classifer(width=args.width, depth=args.depth),
        dm=dm,
        optim=args.optim,
        optim_kwargs={"lr": args.learning_rate},
    )

    summary(classifier_trainer.model, input_size=(args.batch_size, 1, 32, 32))

    logger = WandbLogger(project="mnist-classifier", name="classifier", log_model=True)
    logger.watch(classifier_trainer.model, log="all")
    logger.log_hyperparams(vars(args))

    trainer = Trainer(max_epochs=args.epochs, logger=logger, default_root_dir="logs")
    trainer.fit(classifier_trainer)
    trainer.test(classifier_trainer)


if __name__ == "__main__":
    args = parse_args()
    main(args)
