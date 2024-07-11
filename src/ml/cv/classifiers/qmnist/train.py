from torchvision import transforms
import torch
import torch.nn as nn

from ml_zoo import datamodules, models, trainers

if __name__ == "__main__":
    # Load the QMNIST data
    dm = datamodules.MNISTDataModule(
        "data",
        dataset_params={
            "download": True,
            "transform": transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                ]
            ),
        }
    )
    dm.prepare_data()
    dm.setup()

    # Load the model
    model = models.image.LeNet.from_version(5)

    # Train the model
    trainer = trainers.ClassificationTrainer(max_epochs=5, log=True)
    trainer.fit(model, dm)
    trainer.test(model, dm)
    trainer.save_model(model, "lenet_model.pth")
