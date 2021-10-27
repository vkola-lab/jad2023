#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 15:40:27 2021

@author: cxue2
"""

import torch
import torch.nn as nn



class ImageDecoder(nn.Module):
    
    
    def __init__(self):
        
        super(ImageDecoder, self).__init__()
        
        self.module = nn.Sequential(
            nn.ConvTranspose3d(512, 1, (18, 22, 18)))
    
    
    def forward(self, xs):
        
        xs = torch.unsqueeze(xs, -1)
        xs = torch.unsqueeze(xs, -1)
        xs = torch.unsqueeze(xs, -1)

        return self.module(xs)
            
            

if __name__ == '__main__':
    
    x = torch.rand(2, 512)
    m = ImageDecoder()
    o = m(x)

    print(o.shape)