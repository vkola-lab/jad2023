import torch
import torch.nn as nn
from torch.nn.modules.flatten import Flatten


class BasicConv3d(nn.Module):   
    
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, bias=False
    ):
        
        super().__init__()

        self.module = nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels, kernel_size, stride, padding, bias=bias
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.module(x)
        

class ImageEncoder(nn.Module):    
    
    def __init__(self, n_fil, dim_z):
        
        super().__init__()

        self.module=nn.Sequential(
            BasicConv3d(1,        n_fil,    3, 2, 1, bias=False),
            BasicConv3d(n_fil,    n_fil*2,  3, 2, 1, bias=False),
            BasicConv3d(n_fil*2,  n_fil*4,  3, 2, 1, bias=False),
            BasicConv3d(n_fil*4,  n_fil*8,  3, 2, 1, bias=False),
            BasicConv3d(n_fil*8,  n_fil*16, 3, 2, 1, bias=False),
            BasicConv3d(n_fil*16, dim_z,    1, 1, 0, bias=False),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Conv3d(dim_z, dim_z, 1, 1, 0, bias=True)
        )
        
    def forward(self, x):
        
        o = self.module(x)
        o = o.view(o.shape[0], -1)
        return o


class ImageEncoder_128(nn.Module):    
    
    def __init__(self, n_fil, dim_z):
        
        super().__init__()

        self.module=nn.Sequential(
            BasicConv3d(1,        n_fil,    3, 2, 1, bias=False),
            BasicConv3d(n_fil,    n_fil*2,  3, 2, 1, bias=False),
            BasicConv3d(n_fil*2,  n_fil*4,  3, 2, 1, bias=False),
            BasicConv3d(n_fil*4,  n_fil*8,  3, 2, 1, bias=False),
            BasicConv3d(n_fil*8,  n_fil*16, 3, 2, 1, bias=False),
            BasicConv3d(n_fil*16, dim_z,    1, 1, 0, bias=False),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Conv3d(dim_z, dim_z, 1, 1, 0, bias=True)
        )
        
    def forward(self, x):
        
        o = self.module(x)
        o = o.view(x.shape[0], -1)
        return o


class ImageEncoder_64(nn.Module):    
    
    def __init__(self, n_fil, dim_z):
        
        super().__init__()

        self.module=nn.Sequential(
            BasicConv3d(1,        n_fil,    3, 2, 1, bias=False),
            BasicConv3d(n_fil,    n_fil*2,  3, 2, 1, bias=False),
            BasicConv3d(n_fil*2,  n_fil*4,  3, 2, 1, bias=False),
            BasicConv3d(n_fil*4,  n_fil*8,  3, 2, 1, bias=False),
            BasicConv3d(n_fil*8,  n_fil*16, 3, 2, 1, bias=False),
            BasicConv3d(n_fil*16, dim_z,    1, 1, 0, bias=False),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Conv3d(dim_z, dim_z, 1, 1, 0, bias=True)
        )
        
    def forward(self, x):
        
        o = self.module(x)
        o = o.view(x.shape[0], -1)
        return o


if __name__ == '__main__':
    
    x = torch.rand(2, 1, 128, 128, 128)
    m = ImageEncoder_128(32, 512)
    o = m(x)

    print(o.shape)
    