import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim


segment_size = 1075
channels = [64, 128, 256, 512, 1024, 512, 512]
kernel_size_conv = [5, 5, 5, 3, 3]
kernel_size_pool = [3, 3, 3, 2, 2]
fc_width = 5
class PitchContourDiscriminator(nn.Module):
    def __init__(self):
        super(PitchContourDiscriminator, self).__init__()
        self.convs = nn.ModuleList([])
        self.pools = nn.ModuleList([])
        for i in range(len(kernel_size_conv)):
            self.convs.append(nn.Conv1d(in_channels=1 if i == 0 else channels[i - 1], out_channels=channels[i], kernel_size=kernel_size_conv[i]))
            self.convs.append(nn.Conv1d(in_channels=channels[i], out_channels=channels[i], kernel_size=kernel_size_conv[i]))
            self.pools.append(nn.MaxPool1d(kernel_size=kernel_size_pool[i]))
        self.fc1 = nn.Linear(channels[-3] * fc_width, channels[-2])
        self.fc2 = nn.Linear(channels[-2], channels[-1])
        self.fc3 = nn.Linear(channels[-1], 1)


    def forward(self, x):
        for i in range(len(kernel_size_conv)):
            x = F.dropout(F.leaky_relu(self.convs[i * 2](x)))
            x = F.dropout(F.leaky_relu(self.convs[i * 2 + 1](x)))
            x = self.pools[i](x)
        x = x.view(-1, channels[-3] * fc_width)
        x = F.dropout(F.leaky_relu(self.fc1(x)))
        x = F.dropout(F.leaky_relu(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x))
        return x
