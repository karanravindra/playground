import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from ema_pytorch import EMA
from ml_zoo.datamodules import CIFARDataModule
from torchinfo import summary
from tqdm import tqdm

dm = CIFARDataModule(
    data_dir="data",
    dataset_params={
        "download": True,
        "transform": torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((32, 32)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomRotation(45),
                torchvision.transforms.ToTensor(),
            ]
        ),
    },
    loader_params={
        "batch_size": 64,
        "num_workers": 2,
    },
)
dm.prepare_data()
dm.setup()
trian_loader = dm.train_dataloader()
test_loader = dm.test_dataloader()


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
        self.layers = nn.Sequential(
            *[
                conv_type(in_channels, out_channels, kernel_size, stride, padding)
                if _ == 0
                else conv_type(out_channels, out_channels, kernel_size, stride, padding)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        return self.layers(x)


class Classifer(nn.Module):
    def __init__(self):
        super(Classifer, self).__init__()
        self.backbone = nn.Sequential(
            ConvStack(3, 32, 2),
            nn.MaxPool2d(2),
            ConvStack(32, 64, 2),
            nn.MaxPool2d(2),
            ConvStack(64, 128, 2),
            nn.MaxPool2d(2),
            ConvStack(128, 256, 2),
            nn.MaxPool2d(2),
            ConvStack(25, 256, 2),
        )
        self.classifier = nn.Linear(256 * 2 * 2, 10, bias=False)

    def forward(self, x):
        x = self.backbone(x).view(x.size(0), -1)
        x = self.classifier(x)
        return x


model = Classifer().to("cuda")
summary(
    model,
    input_data=torch.randn(64, 3, 32, 32, device="cuda", requires_grad=False),
)

ema = EMA(model, beta=0.9999, update_after_step=100, update_every=10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
criterion = nn.CrossEntropyLoss()

test_loss = 0
test_acc = 0
for epoch in range(10):
    model.train()
    pbar = tqdm(trian_loader, desc=f"Epoch {epoch+1}")
    for img, label in pbar:
        img, label = img.to("cuda"), label.to("cuda")
        optimizer.zero_grad()
        output = model(img)

        loss = criterion(output, label)

        loss.backward()
        optimizer.step()

        ema.update()
        pbar.set_postfix_str(
            f"loss: {loss.item():.4f}, test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}"
        )

    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for img, label in tqdm(test_loader, desc="Testing", leave=True):
            img, label = img.to("cuda"), label.to("cuda")
            output = model(img)
            test_loss += criterion(output, label)
            test_acc += (output.argmax(1) == label).float().mean()

    test_loss /= len(test_loader)
    test_loss = test_loss.item()
    test_acc /= len(test_loader)
    test_acc = test_acc.item()

torch.save(model.state_dict(), "model.pth")
