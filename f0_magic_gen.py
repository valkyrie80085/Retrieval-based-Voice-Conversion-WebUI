import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

segment_size = 1215
channels = [64, 128, 256, 512, 1024, 2048]
kernel_size_conv = [5, 5, 5, 5, 5]
kernel_size_pool = [3, 3, 3, 3, 3]
bridge_width = 5
class PitchContourGenerator(nn.Module):
    def __init__(self):
        super(PitchContourGenerator, self).__init__()
        self.down_convs = nn.ModuleList([])
        self.up_convs = nn.ModuleList([])

        self.down_idmappings = nn.ModuleList([])
        self.up_idmappings = nn.ModuleList([])

        self.downsamples = nn.ModuleList([])
        self.upsamples = nn.ModuleList([])

        for i in range(len(kernel_size_conv)):
            self.down_convs.append(nn.Conv1d(in_channels=1 if i == 0 else channels[i - 1], out_channels=channels[i], kernel_size=kernel_size_conv[i], padding="same"))
            self.up_convs.append(nn.Conv1d(out_channels=1 if i == 0 else channels[i - 1], in_channels=channels[i], kernel_size=kernel_size_conv[i], padding="same"))

            self.down_idmappings.append(nn.Conv1d(in_channels=1 if i == 0 else channels[i - 1], out_channels=channels[i], kernel_size=1, padding="same"))

            self.down_convs.append(nn.Conv1d(in_channels=channels[i], out_channels=channels[i], kernel_size=kernel_size_conv[i], padding="same"))
            self.up_convs.append(nn.Conv1d(out_channels=channels[i], in_channels=channels[i] * 2, kernel_size=kernel_size_conv[i], padding="same"))

            self.up_idmappings.append(nn.Conv1d(out_channels=1 if i == 0 else channels[i - 1], in_channels=channels[i] * 2, kernel_size=1, padding="same"))

            self.downsamples.append(nn.MaxPool1d(kernel_size=kernel_size_pool[i]))
            self.upsamples.append(nn.Upsample(scale_factor=kernel_size_pool[i]))

        self.bridge_idmappings = nn.Conv1d(channels[-2], channels[-2], 1, padding="same")
        self.bridge = nn.Sequential(
            nn.Conv1d(channels[-2], channels[-1], bridge_width, padding="same"),
            nn.BatchNorm1d(channels[-1]),
            nn.ReLU(),
            nn.Conv1d(channels[-1], channels[-2], bridge_width, padding="same")
        ) 


    def forward(self, x):
        skips = []
        for i in range(len(kernel_size_conv)):
            s = self.down_idmappings[i](x)
            if i > 0:
                x = F.relu(x)
            x = F.relu(self.down_convs[i * 2](x))
            x = self.down_convs[i * 2 + 1](x) + s
            skips.append(x)
            x = self.downsamples[i](x)
        s = self.bridge_idmappings(x)
        x = F.relu(x)
        x = self.bridge(x) + s
        for i in reversed(range(len(kernel_size_conv))):
            x = self.upsamples[i](x)
            x = torch.cat((x, skips[i]), dim=1)
            s = self.up_idmappings[i](x)
            x = F.relu(x)
            x = F.relu(self.up_convs[i * 2 + 1](x))
            x = self.up_convs[i * 2](x) + s
        return x
