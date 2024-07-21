import argparse

import torchvision
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from nn_zoo.datamodules import CIFARDataModule
from nn_zoo.trainers import AutoEncoderTrainer

from model import AutoEncoder

def parse_args():
    params = dict(
        learning_rate=4e-4,
        width=2,
        depth=1,
        optim="adam",
        batch_size=12,
        epochs=10,
        num_workers=4,
        prefetch_factor=None,
        pin_memory=True,
        persistent_workers=True,
    )

    parser = argparse.ArgumentParser(description="Train a CIFAR autoencoder")
    for k, v in params.items():
        parser.add_argument(f"--{k}", type=type(v), default=v)
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

    classifier_trainer = AutoEncoderTrainer(
        model=AutoEncoder(
            width=args.width,
            depth=args.depth,
        ),
        dm=dm,
        optim=args.optim,
        optim_kwargs={"lr": args.learning_rate},
    )

    summary(classifier_trainer.model, input_size=(args.batch_size, 1, 32, 32), depth=2)

    logger = WandbLogger(
        project="mnist-autoencoder", name="autoencoder", log_model=True
    )
    logger.watch(classifier_trainer.model, log="all")
    logger.log_hyperparams(vars(args))

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
    main(args)
