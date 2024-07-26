from torch import nn
from torch.nn import functional as F
from torchinfo import summary
from nn_zoo.models.components import DepthwiseSeparableConv2d, VectorQuantizer
from torchmetrics.functional.image.ssim import structural_similarity_index_measure as ssim_func
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

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
            nn.GroupNorm(in_channels // 4 if in_channels >= 4 else 1, in_channels),
            DepthwiseSeparableConv2d(in_channels, out_channels, 3),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.layers[0](x)
        for i, layer in enumerate(self.layers[1:]):
            x = layer(x) + x
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depth: int):
        super(DownBlock, self).__init__()
        self.block = nn.Sequential(
            Block(in_channels, out_channels, depth),
            nn.MaxPool2d(2),
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
            DepthwiseSeparableConv2d(width * 4, width * 2, 3),
        )
        self.proj_in = nn.Identity()  # nn.Conv2d(width, width, 1)
        self.vq = nn.Identity()
        self.proj_out = nn.Identity()  # nn.Conv2d(width, width, 1)
        self.decoder = nn.Sequential(
            DepthwiseSeparableConv2d(width * 2, width * 4, 3),
            UpBlock(width * 4, width * 4, depth),
            UpBlock(width * 4, width * 2, depth),
            UpBlock(width * 2, width, depth),
            Block(width, 1, depth),
            nn.Sigmoid(),
        )
        
        self.register_module("lpips", LPIPS(net_type="squeeze", normalize=True))

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
    # @staticmethod
    def loss(self, x, y):
        mse = F.mse_loss(x, y)
        bce = F.binary_cross_entropy(x, y)
        psnr = 10 * (1 / mse).log10()
        ssim = ssim_func(x, y)
        lpips = self.lpips(x.repeat(1, 3, 1, 1), y.repeat(1, 3, 1, 1))

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
    
    model = AutoEncoder(width=8, depth=4).cuda()
    
    x1 = torch.rand(1, 1, 32, 32).cuda()# * 2 - 1
    x2 = torch.tanh(model(x1))
    print(x1.shape, x2.shape)
    
    print(f"loss: {model.loss(x2, x1)}")
    
    # summary(
    #     model,
    #     input_size=(512, 1, 32, 32),
    #     depth=2,
    #     col_names=["output_size", "params_percent"],
    # )
