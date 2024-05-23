import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

periods = [1, 2, 3, 5, 7, 11]
segment_size = [1939, 1934, 1929, 1865, 1855, 2024]
depth = [5, 4, 4, 3, 3, 3]
channels = [64, 128, 256, 256, 512]
kernel_size_conv = [5, 5, 5, 5, 5]
kernel_size_pool = [3, 3, 3, 3, 3]
fc_width = [3, 7, 3, 9, 5, 2]
class PitchContourDiscriminatorP(nn.Module):
    def __init__(self, p, t):
        super(PitchContourDiscriminatorP, self).__init__()
        self.p = p
        self.t = t
        self.convs = nn.ModuleList([])
        self.pools = nn.ModuleList([])
        for i in range(depth[self.p]):
            self.convs.append(nn.Conv1d(in_channels=3 if i == 0 else channels[i - 1], out_channels=channels[i], kernel_size=kernel_size_conv[i]))
            self.convs.append(nn.Conv1d(in_channels=channels[i], out_channels=channels[i], kernel_size=kernel_size_conv[i]))
            self.pools.append(nn.MaxPool1d(kernel_size=kernel_size_pool[i]))
        self.fc1 = nn.Linear(channels[depth[self.p] - 1] * fc_width[self.p], channels[depth[self.p] - 1] // 2)
        self.fc2 = nn.Linear(channels[depth[self.p] - 1] // 2, channels[depth[self.p] - 1] // 2)
        self.fc3 = nn.Linear(channels[depth[self.p] - 1] // 2, 1)


    def forward(self, x):
        x = x[:, :, (x.shape[2] + segment_size[self.p]) // 2 - segment_size[self.p]:(x.shape[2] + segment_size[self.p]) // 2]
        x = x.view(x.shape[0], x.shape[1], -1, periods[self.p])
        if self.t:
            x = torch.transpose(x, 2, 3)
        x = torch.transpose(x, 1, 2)
        x = x.reshape(x.shape[0] * periods[self.p], x.shape[2], -1)
        for i in range(depth[self.p]):
            x = F.leaky_relu(self.convs[i * 2](x))
            x = F.leaky_relu(self.convs[i * 2 + 1](x))
            x = self.pools[i](x)
        x = x.view(x.shape[0], -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = x.view(x.shape[0] // periods[self.p], periods[self.p])
        return x


class PitchContourDiscriminator(nn.Module):
    def __init__(self):
        super(PitchContourDiscriminator, self).__init__()
        self.discs = nn.ModuleList([])
        for i in range(len(periods)):
            self.discs.append(PitchContourDiscriminatorP(i, False))
            if periods[i] > 1:
                self.discs.append(PitchContourDiscriminatorP(i, True))


    def forward(self, x):
        return torch.cat(tuple(disc(x) for disc in self.discs), dim=1)
