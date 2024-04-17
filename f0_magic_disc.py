import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

periods = [1, 2, 3, 5, 7] #11
segment_size = [1075, 1070, 1281, 1055, 1288] #1133
depth = [5, 4, 4, 3, 3] #2
channels = [32, 64, 128, 256, 512]
kernel_size_conv = [5, 5, 5, 3, 3]
kernel_size_pool = [3, 3, 3, 2, 2]
fc_width = [5, 5, 3, 3, 2] #7
class PitchContourDiscriminatorP(nn.Module):
    def __init__(self, p):
        super(PitchContourDiscriminatorP, self).__init__()
        self.p = p
        self.convs = nn.ModuleList([])
        self.pools = nn.ModuleList([])
        for i in range(depth[self.p]):
            self.convs.append(nn.Conv1d(in_channels=1 if i == 0 else channels[i - 1], out_channels=channels[i], kernel_size=kernel_size_conv[i]))
            self.convs.append(nn.Conv1d(in_channels=channels[i], out_channels=channels[i], kernel_size=kernel_size_conv[i]))
            self.pools.append(nn.MaxPool1d(kernel_size=kernel_size_pool[i]))
        self.fc1 = nn.Linear(channels[depth[self.p] - 1] * fc_width[self.p], channels[depth[self.p] - 1] // 2)
        self.fc2 = nn.Linear(channels[depth[self.p] - 1] // 2, channels[depth[self.p] - 1] // 2)
        self.fc3 = nn.Linear(channels[depth[self.p] - 1] // 2, 1)


    def forward(self, x):
        x = x[:, :, (x.shape[2] + segment_size[self.p]) // 2 - segment_size[self.p]:(x.shape[2] + segment_size[self.p]) // 2]
        x = x.view(x.shape[0], x.shape[1], -1, periods[self.p])
        x = torch.transpose(x, 2, 3)
        x = x.reshape(x.shape[0] * periods[self.p], x.shape[1], -1)
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
            self.discs.append(PitchContourDiscriminatorP(i))


    def forward(self, x):
        return torch.cat(tuple(disc(x) for disc in self.discs), dim=1)
