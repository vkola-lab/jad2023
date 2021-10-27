#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 15:40:27 2021

@author: cxue2
"""

import torch
import torch.nn as nn



class ImageEncoder(nn.Module):
    
    
    def __init__(self):
        
        super(ImageEncoder, self).__init__()
        
        self.module = nn.Sequential(
            nn.Conv3d(1, 512, 1),
            
            # global average pooling 
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            
            # squeeze
            nn.Flatten(),)
    
    
    def forward(self, xs):
        
        return self.module(xs)
            
            

if __name__ == '__main__':
    
    x = torch.rand(2, 1, 18, 22, 18)
    m = ImageEncoder()
    o = m(x)

    print(o.shape)
    