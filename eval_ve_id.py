#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 14:46:56 2021

@author: cxue2
"""

import torch
import torch.nn as nn

from net_ve import VoiceEncoder
from net_id import ImageDecoder

# load voice encoder and image decoder
net = nn.Sequential(VoiceEncoder(), ImageDecoder())
net[0].load_state_dict(torch.load('./save/exp_1/net_ve.pt'))
net[1].load_state_dict(torch.load('./save/exp_1/net_id.pt'))

# evaluate
pass