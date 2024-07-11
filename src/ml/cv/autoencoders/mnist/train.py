import torch
import torch.nn as nn

from ml_zoo import datamodules, models, trainers

if __name__ == "__main__":
    # Load the QMNIST data
    dm = datamodules.MNISTDataModule()
    dm.prepare_data()
    dm.setup()

    # Load the model
    model = models.Autoencoder()

    # Train the model
    trainer = trainers.AutoencoderTrainer(max_epochs=5, log=True)
    trainer.fit(model, dm)
    trainer.test(model, dm)
    trainer.save_model(model, "autoencoder_model.pth")