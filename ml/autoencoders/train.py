import argparse
import os

import torchvision
from torchinfo import summary
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from nn_zoo.trainers import AutoEncoderTrainer


def parse_args():
    params = dict(
        model=None,
        learning_rate=4e-4,
        width=2,
        depth=1,
        optim="adam",
        batch_size=128,
        epochs=10,
        num_workers=4,
        prefetch_factor=None,
        pin_memory=True,
        persistent_workers=True,
    )

    parser = argparse.ArgumentParser(description="Train an autoencoder")
    parser.add_argument("--model", type=str, required=True, help="Model to train")
    parser.add_argument("--learning_rate", type=float, default=params["learning_rate"])
    parser.add_argument("--width", type=int, default=params["width"])
    parser.add_argument("--depth", type=int, default=params["depth"])
    parser.add_argument("--optim", type=str, default=params["optim"])
    parser.add_argument("--batch_size", type=int, default=params["batch_size"])
    parser.add_argument("--epochs", type=int, default=params["epochs"])
    parser.add_argument("--num_workers", type=int, default=params["num_workers"])
    parser.add_argument(
        "--prefetch_factor", type=int, default=params["prefetch_factor"]
    )
    parser.add_argument("--pin_memory", type=bool, default=params["pin_memory"])
    parser.add_argument(
        "--persistent_workers", type=bool, default=params["persistent_workers"]
    )
    return parser.parse_args()


def main(args, ModelType, DataModuleType):
    dm = DataModuleType(
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
        model=ModelType.AutoEncoder(
            width=args.width,
            depth=args.depth,
        ),
        dm=dm,
        optim=args.optim,
        optim_kwargs={"lr": args.learning_rate},
    )

    logger = WandbLogger(
        project="mnist-autoencoder", name="autoencoder", log_model=True
    )
    logger.watch(classifier_trainer.model, log="all")
    logger.log_hyperparams(vars(args))
    logger.log_hyperparams(
        {"num_params": sum(p.numel() for p in classifier_trainer.model.parameters())}
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        logger=logger,
        default_root_dir="logs",
        accumulate_grad_batches=1,
        check_val_every_n_epoch=1,
    )
    trainer.fit(classifier_trainer)
    trainer.test(classifier_trainer)


if __name__ == "__main__":
    args = parse_args()
    assert (
        args.model in os.listdir(os.path.dirname(os.path.realpath(__file__)))
    ), f"Model {args.model} not found in {os.listdir(os.path.dirname(os.path.realpath(__file__)))}"

    # import requested module
    ModelType = __import__(args.model)
    print(ModelType)

    # import datamodule
    if args.model.lower() == "mnist":
        from nn_zoo.datamodules import MNISTDataModule

        DataModuleType = MNISTDataModule
    elif args.model.lower() == "cifar":
        from nn_zoo.datamodules import CIFARDataModule

        DataModuleType = CIFARDataModule

    main(args, ModelType, DataModuleType)
