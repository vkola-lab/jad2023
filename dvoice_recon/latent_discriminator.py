import torch.nn as nn


class LatentDiscriminator(nn.Module):

    def __init__(self, dim_z):

        super().__init__()

        self.module = nn.Sequential(
            nn.Linear(dim_z, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):

        return self.module(x)


if __name__ == '__main__':

    import torch

    dim_z = 32
    net = LatentDiscriminator(dim_z=32)
    x = torch.zeros((3, dim_z), dtype=torch.float32)

    print(net(x).shape)