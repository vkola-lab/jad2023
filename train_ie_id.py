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

from dataset_image import ImageDataset

# multi-gpu framework
from xfdlfw import Model
from xfdlfw.metric import MeanSquaredError


if __name__ == '__main__':

    # initialize net/model
    net = nn.Sequential(ImageEncoder(), ImageDecoder())
    mdl = Model(net)

    # initialize datasets
    selections = {
        # 'dataset': ['ADNI1', 'ADNI2', 'ADNI3', 'ADNIGO', 'NACC_ALL'],
        'dataset': ['NACC_ALL'],
    } 
    dst_trn = ImageDataset('./data/mri_ayan.csv', selections, 0, [4, 1])
    dst_vld = ImageDataset('./data/mri_ayan.csv', selections, 1, [4, 1])

    # training dataloader parameters
    kwargs_ldr_trn = {
        'dataset': dst_trn,
        'batch_size': 4,
        'shuffle': True,
        'num_workers': 1,
    }

    # validation dataloader parameters
    kwargs_ldr_vld = {
        'dataset': dst_vld,
        'batch_size': 4,
        'shuffle': False,
        'num_workers': 1
    }

    # loss and optimizer
    losses = [
        torch.nn.MSELoss()
    ]

    optimizers = [
        torch.optim.Adam(net.parameters(), lr=1e-4),
    ]

    # devices for training
    devices = [0, 1, 2, 3]

    # train image encoder and decoder
    mdl.fit(
        kwargs_ldr_trn,
        losses, optimizers, devices,
        n_epochs=128,
        kwargs_ldr_vld=kwargs_ldr_vld,
        metrics_disp=[MeanSquaredError()],
        metrics_crit=[MeanSquaredError()],
        save_mode = 0,
        # save_dir = './checkpoints'
    )

    # save nets
    torch.save({k: v.cpu() for k, v in net[0].state_dict().items()}, './save/exp_1/net_ie_test.pt')
    torch.save({k: v.cpu() for k, v in net[1].state_dict().items()}, './save/exp_1/net_id_test.pt')