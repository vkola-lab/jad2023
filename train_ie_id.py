#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 11:09:36 2021

@author: cxue2
"""

import torch
import torch.nn as nn

from net_ie import ImageEncoder
from net_id import ImageDecoder



# new nets
net = nn.Sequential(ImageEncoder(), ImageDecoder())

# train image encoder and decoder
pass

# save nets
torch.save({k: v.cpu() for k, v in net[0].state_dict().items()}, './save/exp_1/net_ie.pt')
torch.save({k: v.cpu() for k, v in net[1].state_dict().items()}, './save/exp_1/net_id.pt')