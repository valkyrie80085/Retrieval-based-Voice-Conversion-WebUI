import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

rescale = 1.0

segment_size = 2187
channels = [64, 128, 256, 512, 512, 1024, 1024]
kernel_size_conv = [5, 5, 5, 5, 5, 5]
kernel_size_pool = [3, 3, 3, 3, 3, 3]
bridge_width = 3


class PitchContourGenerator(nn.Module):
    def __init__(self, c=2):
        super(PitchContourGenerator, self).__init__()
        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        self.down_idmappings = nn.ModuleList([])
        self.up_idmappings = nn.ModuleList([])

        self.downsamples = nn.ModuleList([])
        self.upsamples = nn.ModuleList([])

        for i in range(len(kernel_size_conv)):
            self.down_idmappings.append(
                nn.Conv1d(
                    in_channels=c if i == 0 else channels[i - 1],
                    out_channels=channels[i],
                    kernel_size=1,
                    padding="same",
                )
            )
            if i == 0:
                self.down_blocks.append(
                    nn.Sequential(
                        nn.Conv1d(
                            in_channels=c,
                            out_channels=channels[i],
                            kernel_size=kernel_size_conv[i],
                            padding="same",
                        ),
                        nn.GELU(),
                        nn.Conv1d(
                            in_channels=channels[i],
                            out_channels=channels[i],
                            kernel_size=kernel_size_conv[i],
                            padding="same",
                        ),
                    )
                )
            else:
                self.down_blocks.append(
                    nn.Sequential(
                        nn.GELU(),
                        nn.Conv1d(
                            in_channels=channels[i - 1],
                            out_channels=channels[i],
                            kernel_size=kernel_size_conv[i],
                            padding="same",
                        ),
                        nn.GELU(),
                        nn.Conv1d(
                            in_channels=channels[i],
                            out_channels=channels[i],
                            kernel_size=kernel_size_conv[i],
                            padding="same",
                        ),
                    )
                )

            self.down_idmappings.append(
                nn.Conv1d(
                    in_channels=channels[i],
                    out_channels=channels[i],
                    kernel_size=1,
                    padding="same",
                )
            )
            self.down_blocks.append(
                nn.Sequential(
                    nn.GELU(),
                    nn.Conv1d(
                        in_channels=channels[i],
                        out_channels=channels[i],
                        kernel_size=kernel_size_conv[i],
                        padding="same",
                    ),
                    nn.GELU(),
                    nn.Conv1d(
                        in_channels=channels[i],
                        out_channels=channels[i],
                        kernel_size=kernel_size_conv[i],
                        padding="same",
                    ),
                )
            )

            self.up_idmappings.append(
                nn.Conv1d(
                    in_channels=channels[i],
                    out_channels=1 if i == 0 else channels[i - 1],
                    kernel_size=1,
                    padding="same",
                )
            )
            self.up_blocks.append(
                nn.Sequential(
                    nn.GELU(),
                    nn.Conv1d(
                        in_channels=channels[i],
                        out_channels=channels[i],
                        kernel_size=kernel_size_conv[i],
                        padding="same",
                    ),
                    nn.GELU(),
                    nn.Conv1d(
                        in_channels=channels[i],
                        out_channels=1 if i == 0 else channels[i - 1],
                        kernel_size=kernel_size_conv[i],
                        padding="same",
                    ),
                )
            )

            self.up_idmappings.append(
                nn.Conv1d(
                    in_channels=2 * channels[i],
                    out_channels=channels[i],
                    kernel_size=1,
                    padding="same",
                )
            )
            self.up_blocks.append(
                nn.Sequential(
                    nn.GELU(),
                    nn.Conv1d(
                        in_channels=2 * channels[i],
                        out_channels=channels[i],
                        kernel_size=kernel_size_conv[i],
                        padding="same",
                    ),
                    nn.GELU(),
                    nn.Conv1d(
                        in_channels=channels[i],
                        out_channels=channels[i],
                        kernel_size=kernel_size_conv[i],
                        padding="same",
                    ),
                )
            )

            self.downsamples.append(nn.MaxPool1d(kernel_size=kernel_size_pool[i]))
            self.upsamples.append(nn.Upsample(scale_factor=kernel_size_pool[i]))

        self.bridge_idmappings1 = nn.Conv1d(
            channels[-2], channels[-1], 1, padding="same"
        )

        self.bridge1 = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(channels[-2], channels[-1], bridge_width, padding="same"),
            nn.GELU(),
            nn.Conv1d(channels[-1], channels[-1], bridge_width, padding="same"),
        )

        self.bridge_idmappings2 = nn.Conv1d(
            channels[-1], channels[-2], 1, padding="same"
        )
        self.bridge2 = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(channels[-1], channels[-1], bridge_width, padding="same"),
            nn.GELU(),
            nn.Conv1d(channels[-1], channels[-2], bridge_width, padding="same"),
        )

        with torch.no_grad():
            for block in (
                self.down_blocks + self.up_blocks + [self.bridge1, self.bridge2]
            ):
                block[-1].weight *= rescale

    def forward(self, x):
        skips = []
        for i in range(len(kernel_size_conv)):
            s = self.down_idmappings[i * 2](x)
            x = self.down_blocks[i * 2](x) + s

            s = self.down_idmappings[i * 2 + 1](x)
            x = self.down_blocks[i * 2 + 1](x) + s

            skips.append(x)
            x = self.downsamples[i](x)

        s = self.bridge_idmappings1(x)
        x = self.bridge1(x) + s

        s = self.bridge_idmappings2(x)
        x = self.bridge2(x) + s

        for i in reversed(range(len(kernel_size_conv))):
            x = self.upsamples[i](x)
            x = torch.cat((x, skips[i]), dim=1)

            s = self.up_idmappings[i * 2 + 1](x)
            x = self.up_blocks[i * 2 + 1](x) + s

            s = self.up_idmappings[i * 2](x)
            x = self.up_blocks[i * 2](x) + s

        return x
