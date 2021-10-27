#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 11:09:36 2021

@author: cxue2
"""

import torch
import torch.nn as nn

from net_ie import ImageEncoder
from net_ve import VoiceEncoder



# load image encoder
net_ie = ImageEncoder()
net_ie.load_state_dict(torch.load('./save/exp_1/net_ie.pt'))

# generate feature/latent vectors using image encoder
# don't forget to DETACH the outputs using Tensor.detach()
pass

# new voice encoder
net_ve = VoiceEncoder()

# train voice encoder
pass

# save
torch.save({k: v.cpu() for k, v in net_ve.state_dict().items()}, './save/exp_1/net_ve.pt')