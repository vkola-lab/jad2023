#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 15:40:27 2021

@author: cxue2
"""

import torch
import torch.nn as nn



class ImageEncoder(nn.Module):
    
    
#    def __init__(self):
#        
#        super(ImageEncoder, self).__init__()
#        
#        self.module = nn.Sequential(
#            nn.Conv3d(1, 512, 1),
#            
#            # global average pooling 
#            nn.AdaptiveAvgPool3d((1, 1, 1)),
#            
#            # squeeze
#            nn.Flatten(),)
#    
#    
#    def forward(self, xs):
#        
#        return self.module(xs)
    
    
    def __init__(self, in_size=1, fil_num=32, out_channels=1):
        
        super().__init__()
        
        self.in_size = in_size
        self.conv1 = nn.Conv3d(in_size, fil_num, 5, 3, 2, bias=False)
        self.bn1 = nn.BatchNorm3d(fil_num)
        self.conv2 = nn.Conv3d(fil_num, 4*fil_num, 5, 3, 2, bias=False)
        self.bn2 = nn.BatchNorm3d(4*fil_num)
        self.conv3 = nn.Conv3d(4*fil_num, 16*fil_num, 5, 3, 2, bias=False)
        self.bn3 = nn.BatchNorm3d(16*fil_num)
        self.conva = nn.LeakyReLU()
        self.avg_pool=nn.AvgPool3d((7,8,7))
        self.flatten=nn.Flatten()

        self.module=nn.Sequential(
                self.conv1,
                self.bn1,
                self.conva,
                self.conv2,
                self.bn2,
                self.conva,
                self.conv3,
                self.bn3,
                self.conva,
                self.avg_pool,
                self.flatten)
        
        
    def forward(self, x):
              
        return self.module(x)
            
            

if __name__ == '__main__':
    
    x = torch.rand(2, 1, 189, 216, 189)
    m = ImageEncoder()
    o = m(x)

    print(o.shape)
    