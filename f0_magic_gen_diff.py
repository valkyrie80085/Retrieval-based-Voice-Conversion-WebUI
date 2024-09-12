import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import math

rescale = 1.0

segment_size = 2187
channels = [64, 128, 256, 512, 512, 1024, 1024]
kernel_size_conv = [5, 5, 5, 5, 5, 5]
kernel_size_pool = [3, 3, 3, 3, 3, 3]
bridge_width = 3

fourier_dim = 64
time_dim = 256


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class resBlock(nn.Module):
    def __init__(self, inc, midc, outc, kernel_size, time_dim, is_initial=False):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=inc, out_channels=midc, kernel_size=kernel_size, padding="same"
        )
        self.conv2 = nn.Conv1d(
            in_channels=midc, out_channels=outc, kernel_size=kernel_size, padding="same"
        )
        self.idmapping = nn.Conv1d(
            in_channels=inc, out_channels=outc, kernel_size=1, padding="same"
        )
        self.is_initial = is_initial
        self.mlp = nn.Linear(time_dim, midc * 2)
        with torch.no_grad():
            self.conv2.weight *= rescale

    def forward(self, x, t):
        s = self.idmapping(x)
        if not self.is_initial:
            x = F.gelu(x)
        x = self.conv1(x)
        scale, shift = self.mlp(t).unsqueeze(2).chunk(2, dim=1)
        x = x * (scale + 1) + shift
        x = F.gelu(x)
        x = self.conv2(x)
        return x + s


class PitchContourGenerator(nn.Module):
    def __init__(self, c=3):
        super(PitchContourGenerator, self).__init__()
        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        self.downsamples = nn.ModuleList([])
        self.upsamples = nn.ModuleList([])

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(fourier_dim),
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        for i in range(len(kernel_size_conv)):
            self.down_blocks.append(
                resBlock(
                    c if i == 0 else channels[i - 1],
                    channels[i],
                    channels[i],
                    kernel_size_conv[i],
                    time_dim,
                    i == 0,
                )
            )
            self.down_blocks.append(
                resBlock(
                    channels[i], channels[i], channels[i], kernel_size_conv[i], time_dim
                )
            )

            self.up_blocks.append(
                resBlock(
                    channels[i],
                    channels[i],
                    1 if i == 0 else channels[i - 1],
                    kernel_size_conv[i],
                    time_dim,
                )
            )
            self.up_blocks.append(
                resBlock(
                    2 * channels[i],
                    channels[i],
                    channels[i],
                    kernel_size_conv[i],
                    time_dim,
                )
            )

            self.downsamples.append(nn.MaxPool1d(kernel_size=kernel_size_pool[i]))
            self.upsamples.append(nn.Upsample(scale_factor=kernel_size_pool[i]))

        self.bridge1 = resBlock(
            channels[-2], channels[-1], channels[-1], bridge_width, time_dim
        )
        self.bridge2 = resBlock(
            channels[-1], channels[-1], channels[-2], bridge_width, time_dim
        )

    def forward(self, x, t):
        t = self.time_mlp(t)
        skips = []
        for i in range(len(kernel_size_conv)):
            x = self.down_blocks[i * 2](x, t)
            x = self.down_blocks[i * 2 + 1](x, t)

            skips.append(x)
            x = self.downsamples[i](x)

        x = self.bridge1(x, t)
        x = self.bridge2(x, t)

        for i in reversed(range(len(kernel_size_conv))):
            x = self.upsamples[i](x)
            x = torch.cat((x, skips[i]), dim=1)

            x = self.up_blocks[i * 2 + 1](x, t)
            x = self.up_blocks[i * 2](x, t)

        return x
