import torch
from torch import nn
from torch.nn import functional as F
from torchinfo import summary
from nn_zoo.models.components import DepthwiseSeparableConv2d


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int):
        super(Block, self).__init__()
        self.layers = nn.ModuleList(
            [
                self._block(in_channels, out_channels)
                if i == 0
                else self._block(out_channels, out_channels)
                for i in range(num_layers)
            ]
        )

    def _block(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            DepthwiseSeparableConv2d(in_channels, out_channels, 3),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(out_channels // 4 if out_channels >= 4 else 1, out_channels),
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
            DepthwiseSeparableConv2d(width, 1, 3),
            nn.Sigmoid(),
        )

    def forward(self, x, y=None):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    @classmethod
    def loss(cls, x, y):
        return F.binary_cross_entropy(x, y)

if __name__ == "__main__":
    model = AutoEncoder(width=2, depth=1)
    summary(model, input_size=(1, 1, 32, 32), depth=2)

