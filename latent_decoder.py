import torch
import torch.nn as nn
from numpy import prod


class BasicDeconv3d(nn.Module):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=False
    ):

        super().__init__()

        self.module = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size, stride, padding, output_padding
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):

        return self.module(x)


class LatentDecoder(nn.Module):    
    
    def __init__(self, n_fil, dim_z, view_z=(6, 7, 6)):
        
        super().__init__()
  
        self.module = nn.Sequential(
            # expand latent feature vector to 3D (corresponding to avg pooling)
            nn.Linear(dim_z, dim_z * prod(view_z)),
            nn.Unflatten(1, (dim_z,) + view_z),
            nn.BatchNorm3d(dim_z),
            nn.ReLU(inplace=False),

            BasicDeconv3d(dim_z,    n_fil*16, 1, 1, 0, 0, bias=False),
            BasicDeconv3d(n_fil*16, n_fil*8,  3, 2, 1, 1, bias=False),
            BasicDeconv3d(n_fil*8,  n_fil*4,  3, 2, 1, (0, 1, 0), bias=False),
            BasicDeconv3d(n_fil*4,  n_fil*2,  3, 2, 1, (1, 0, 1), bias=False),
            BasicDeconv3d(n_fil*2,  n_fil*1,  3, 2, 1, 0, bias=False),
            BasicDeconv3d(n_fil*1,  1,        3, 2, 1, 1, bias=False),

            nn.Conv3d(1, 1, 1, 1, 0, bias=True),
        )


    def forward(self, x):
        
        return self.module(x)


class LatentDecoder_128(nn.Module):    
    
    def __init__(self, n_fil, dim_z, view_z=(4, 4, 4)):
        
        super().__init__()
  
        self.module = nn.Sequential(
            # expand latent feature vector to 3D (corresponding to avg pooling)
            nn.Linear(dim_z, dim_z * prod(view_z)),
            nn.Unflatten(1, (dim_z,) + view_z),
            nn.BatchNorm3d(dim_z),
            nn.ReLU(inplace=False),

            BasicDeconv3d(dim_z,    n_fil*16, 1, 1, 0, 0, bias=False),
            BasicDeconv3d(n_fil*16, n_fil*8,  3, 2, 1, 1, bias=False),
            BasicDeconv3d(n_fil*8,  n_fil*4,  3, 2, 1, 1, bias=False),
            BasicDeconv3d(n_fil*4,  n_fil*2,  3, 2, 1, 1, bias=False),
            BasicDeconv3d(n_fil*2,  n_fil*1,  3, 2, 1, 1, bias=False),
            BasicDeconv3d(n_fil*1,  1,        3, 2, 1, 1, bias=False),

            nn.Conv3d(1, 1, 1, 1, 0, bias=True),
        )


    def forward(self, x):
        
        return self.module(x)


class LatentDecoder_64(nn.Module):    
    
    def __init__(self, n_fil, dim_z, view_z=(2, 2, 2)):
        
        super().__init__()
  
        self.module = nn.Sequential(
            # expand latent feature vector to 3D (corresponding to avg pooling)
            nn.Linear(dim_z, dim_z * prod(view_z)),
            # nn.Unflatten(1, (dim_z,) + view_z),
            nn.Unflatten(-1, (dim_z,) + view_z),
            # nn.Unflatten((dim_z,) + view_z),
            nn.BatchNorm3d(dim_z),
            nn.ReLU(inplace=False),

            BasicDeconv3d(dim_z,    n_fil*16, 1, 1, 0, 0, bias=False),
            BasicDeconv3d(n_fil*16, n_fil*8,  3, 2, 1, 1, bias=False),
            BasicDeconv3d(n_fil*8,  n_fil*4,  3, 2, 1, 1, bias=False),
            BasicDeconv3d(n_fil*4,  n_fil*2,  3, 2, 1, 1, bias=False),
            BasicDeconv3d(n_fil*2,  n_fil*1,  3, 2, 1, 1, bias=False),
            BasicDeconv3d(n_fil*1,  1,        3, 2, 1, 1, bias=False),

            nn.Conv3d(1, 1, 1, 1, 0, bias=True),
        )


    def forward(self, x):
        
        return self.module(x)            
            

if __name__ == '__main__':
    
    x = torch.rand(2, 512)
    m = LatentDecoder_128(32, 512, (4, 4, 4))
    o = m(x)

    print(o.shape)