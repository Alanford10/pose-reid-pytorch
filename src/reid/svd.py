# coding=utf-8
# Created by reid_demo on 2019-03-12 10:10
# Copyright © 2019 Alan. All rights reserved.

# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
from src.reid.model import ft_net, ft_net_dense
import yaml


# Load model
def load_network(network, save_path = './models/ft_ResNet50/svd_model.pth'):
    network.load_state_dict(torch.load(save_path))
    return network


# Extract feature
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip


def extract_feature(model,img,use_dense=False):

    features = torch.FloatTensor()
    count = 0
    # numpy image: H x W x C
    # torch image: C x H x W
    # np.transpose( xxx,  (2, 0, 1))
    img = torch.from_numpy(np.transpose(img, (0, 3, 1, 2)))
    img = img.type(torch.FloatTensor)
    n, c, h, w = img.size()
    count += n

    if use_dense:
        ff = torch.FloatTensor(n,1024).zero_()
    else:
        ff = torch.FloatTensor(n,2048).zero_()
    for i in range(2):
        if(i==1):
            img = fliplr(img)
        input_img = Variable(img.cuda())
        outputs = model(input_img)
        # print(outputs.shape)
        f = outputs.data.cpu().float()
        ff = ff+f
    # norm feature

    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    ff = ff.div(fnorm.expand_as(ff))
    features = torch.cat((features,ff), 0)

    return features.numpy()
