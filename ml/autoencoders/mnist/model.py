from torch import nn
from torch.nn import functional as F
from torchinfo import summary
from nn_zoo.models.components import DepthwiseSeparableConv2d, VectorQuantizer
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_func
import lpips

import warnings

warnings.filterwarnings("ignore")


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
            nn.GroupNorm(1, in_channels),
            nn.GELU(),
            DepthwiseSeparableConv2d(in_channels, out_channels, 3),
        )

    def forward(self, x):
        x = self.layers[0](x)
        for i, layer in enumerate(self.layers[1:]):
            x = layer(x) + x
        return x


class DownBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, depth: int):
        super(DownBlock, self).__init__(
            Block(in_channels * 4, out_channels, depth),
            # nn.MaxPool2d(2)
            nn.PixelUnshuffle(2),
        )


class UpBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, depth: int):
        super(UpBlock, self).__init__()
        self.block = nn.Sequential(
            nn.PixelShuffle(2),
            # nn.Upsample(scale_factor=2, mode="nearest"),
            Block(in_channels, out_channels * 4, depth),
        )

    def forward(self, x):
        return self.block(x)


class AutoEncoder(nn.Module):
    def __init__(self, width: int, depth: int):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            Block(1, width * 4, depth),
            DownBlock(width, width, depth),
            DownBlock(width, width, depth),
            DownBlock(width, width, depth),
            DepthwiseSeparableConv2d(width * 4, width * 2, 1, padding=0),
        )
        self.proj_in = nn.Identity()
        self.vq = nn.Identity()
        # VectorQuantizer(width, 8, use_ema=True, decay=0.99, epsilon=1e-5)
        self.proj_out = nn.Identity()  # nn.Conv2d(width, width, 1)
        self.decoder = nn.Sequential(
            DepthwiseSeparableConv2d(width * 2, width * 4, 1, padding=0),
            UpBlock(width, width, depth),
            UpBlock(width, width, depth),
            UpBlock(width, width, depth),
            Block(width * 4, 1, depth),
            nn.Tanh(),
        )

        self.register_module(
            "lpips", lpips.LPIPS(net="squeeze", verbose=False, lpips=False)
        )

    def encode(self, x):
        x = self.encoder(x)
        x = self.proj_in(x)
        return self.vq(x)  # quant_x, dict_loss, commit_loss, indices = self.vq(x)

    def decode(self, x):
        x = self.proj_out(x)
        x = self.decoder(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    # @classmethod
    def loss(self, x, y):
        mse = F.mse_loss(x, y)
        bce = F.binary_cross_entropy((x + 1) / 2, (y + 1) / 2)
        psnr = 10 * (1 / mse).log10()
        ssim = ssim_func(x, y)
        lpips = self.lpips(x.repeat(1, 3, 1, 1), y.repeat(1, 3, 1, 1)).mean()

        return {
            "loss": bce + lpips,
            "bce": bce,
            "mse": mse,
            "ssim": ssim,
            "psnr": psnr,
            "lpips": lpips,
        }


if __name__ == "__main__":
    import torch
    
    model = AutoEncoder(width=4, depth=4).cuda()
    input_tensor = torch.rand(512, 1, 32, 32).cuda()
    output = model(input_tensor)
    print(model.loss(output, input_tensor))
    summary(
        model,
        input_size=(512, 1, 32, 32),
        depth=2,
        col_names=["output_size", "params_percent"],
    )
