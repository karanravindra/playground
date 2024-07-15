import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from lightning import LightningDataModule

__all__ = ["ClassifierTrainer", "ConvStack"]

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.norm1 = nn.GroupNorm(4, out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.act(x)

        x = self.pointwise(x)
        x = self.norm1(x)
        x = self.act(x)

        return x


class ConvStack(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_layers,
        kernel_size=3,
        stride=1,
        padding=1,
        conv_type=DepthwiseSeparableConv2d,
    ):
        super(ConvStack, self).__init__()
        self.layers = nn.ModuleList(
            [
                conv_type(in_channels, out_channels, kernel_size, stride, padding)
                if _ == 0
                else conv_type(out_channels, out_channels, kernel_size, stride, padding)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        x = self.layers[0](x)
        
        for layer in self.layers[1:]:
            residual = x
            x = layer(x) + residual
            
        return x


def get_optim(
    optim: str,
) -> type[torch.optim.SGD | torch.optim.Adam | torch.optim.AdamW]:
    match optim.lower():
        case "sgd":
            return torch.optim.SGD
        case "adam":
            return torch.optim.Adam
        case "adamw":
            return torch.optim.AdamW
        case _:
            raise NotImplementedError(
                f"The requested optimizer: {optim} is not availible"
            )


def get_scheduler(
    scheduler: str,
) -> type[
    torch.optim.lr_scheduler.StepLR
    | torch.optim.lr_scheduler.MultiStepLR
    | torch.optim.lr_scheduler.ExponentialLR
    | torch.optim.lr_scheduler.CosineAnnealingLR
]:
    match scheduler.lower():
        case "steplr":
            return torch.optim.lr_scheduler.StepLR
        case "multisteplr":
            return torch.optim.lr_scheduler.MultiStepLR
        case "exponentiallr":
            return torch.optim.lr_scheduler.ExponentialLR
        case "cosinelr":
            return torch.optim.lr_scheduler.CosineAnnealingLR
        case None:
            return None
        case _:
            raise NotImplementedError(
                f"The requested scheduler: {scheduler} is not availible"
            )


class ClassifierTrainer(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        dm: LightningDataModule,
        optim: str,
        optim_kwargs: dict,
        scheduler: str | None = None,
        scheduler_args: dict | None = None,
    ):
        super(ClassifierTrainer, self).__init__()
        self.model = model
        self.dm = dm
        self.optim = get_optim(optim)
        self.optim_kwargs = optim_kwargs
        self.scheduler = get_scheduler(scheduler) if scheduler else None
        self.scheduler_kwargs = scheduler_args if scheduler else None

    def forward(self, x):
        return self.model(x).output

    def training_step(self, batch, batch_idx):
        x, y = batch

        out = self.model(x, y)
        _, loss = out

        self.log("train_loss", loss)
        self.log("train_acc", (out.argmax(1) == y).float().mean())

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        out = self.model(x, y)
        _, loss = out

        self.log("val_loss", loss)
        self.log("val_acc", (out.argmax(1) == y).float().mean())

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        out = self.model(x, y)
        _, loss = out

        self.log("test_loss", loss)
        self.log("test_acc", (out.argmax(1) == y).float().mean())

        return loss

    def configure_optimizers(self):
        optimizer = self.optim(self.model.parameters(), **self.optim_kwargs)
        scheduler = (
            self.scheduler(optimizer, **self.scheduler_kwargs)
            if self.scheduler
            else None
        )

        if scheduler:
            return [optimizer], [scheduler]
        else:
            return optimizer

    def prepare_data(self) -> None:
        self.dm.prepare_data()

    def setup(self, stage):
        if stage == "fit":
            self.dm.setup("fit")
        elif stage == "test":
            self.dm.setup("test")

    def train_dataloader(self):
        return self.dm.train_dataloader()

    def val_dataloader(self):
        return self.dm.val_dataloader()

    def test_dataloader(self):
        return self.dm.test_dataloader()
