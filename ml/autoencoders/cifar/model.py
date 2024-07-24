from torch import nn
from torch.nn import functional as F
from torchinfo import summary
from nn_zoo.models.components import DepthwiseSeparableConv2d, VectorQuantizer


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
            nn.GroupNorm(in_channels // 4 if in_channels >= 4 else 1, in_channels),
            DepthwiseSeparableConv2d(in_channels, out_channels, 3),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.layers[0](x)
        for layer in self.layers[1:]:
            x = x + layer(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depth: int):
        super(DownBlock, self).__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            Block(in_channels, out_channels, depth),
        )

    def forward(self, x):
        return self.block(x)
    
class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depth: int):
        super(UpBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Block(in_channels, out_channels, depth),
        )

    def forward(self, x):
        return self.block(x)

class AutoEncoder(nn.Module):
    def __init__(self, width: int, depth: int):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            Block(1, width, depth),
            DownBlock(width, width * 2, depth),
            DownBlock(width * 2, width * 4, depth),
            DownBlock(width * 4, width * 4, depth),
            DepthwiseSeparableConv2d(width * 4, width, 3),
        )
        # self.vq = VectorQuantizer(width, 8, use_ema=True, decay=0.99, epsilon=1e-5)
        self.vq = nn.Identity()
        self.decoder = nn.Sequential(
            DepthwiseSeparableConv2d(width, width * 4, 3),
            UpBlock(width * 4, width * 4, depth),
            UpBlock(width * 4, width * 2, depth),
            UpBlock(width * 2, width, depth),
            Block(width, 1, depth),
            nn.Sigmoid(),
        )

    def forward(self, x, y=None):
        x = self.encoder(x)
        # quant_x, dict_loss, commit_loss, indices = self.vq(x)
        quant_x = self.vq(x)
        x = self.decoder(quant_x)
        return x

    @classmethod
    def loss(cls, x, y):
        return F.binary_cross_entropy(x, y)


if __name__ == "__main__":
    model = AutoEncoder(width=4, depth=4)
    summary(model, input_size=(64, 1, 32, 32), depth=2, col_names=["output_size", "params_percent"])
