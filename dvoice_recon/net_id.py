#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 15:40:27 2021

@author: cxue2
"""

import torch
import torch.nn as nn



class ImageDecoder(nn.Module):    
    
    def __init__(self, in_size=1, fil_num=32, out_channels=1):
        
        super().__init__()
        
        self.in_size = in_size
        self.conva = nn.LeakyReLU()
        self.deconv1 = nn.ConvTranspose3d(16*fil_num,4*fil_num,5,3,1,output_padding=0)
        self.bn4 = nn.BatchNorm3d(4*fil_num)
        self.deconv2 = nn.ConvTranspose3d(4*fil_num,fil_num,5,3,1,output_padding=0)
        self.bn5 = nn.BatchNorm3d(fil_num)
        self.deconv3 = nn.ConvTranspose3d(fil_num,self.in_size,5,3,1,output_padding=0)
        self.bn6 = nn.BatchNorm3d(self.in_size)
        self.reverse_linear = nn.Linear(512, 512 * 7 * 8 * 7) #this is the layer that will build up o
        self.avg_pool = nn.AvgPool3d((7, 8, 7))
        self.flatten = nn.Flatten()
  
        self.module = nn.Sequential(
                self.deconv1,
                self.bn4,
                self.conva,
                self.deconv2,
                self.bn5,
                self.conva,
                self.deconv3,
                self.bn6,
                self.conva)


    def forward(self, x):
              
        #now get latent vector back up to a size that is suitable for the decoder portion
        x=self.reverse_linear(x)
                
        #now get latent representation back up to a SHAPE that is suitable for decoder portion
        x=x.view(-1, 512, 7, 8, 7)
        
        #decoder block
        x=self.module(x)
        
        return x
            
            

if __name__ == '__main__':
    
    x = torch.rand(2, 512)
    m = ImageDecoder()
    o = m(x)

    print(o.shape)