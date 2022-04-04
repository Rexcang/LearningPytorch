#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/4/4 19:02
# @Author  : Cheng liwei
# @FileName: Resnet18.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F

class basic_block(nn.Module):

    def  __init__(self,in_channels):
        super(basic_block,self).__init__()
        self.conv1=nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1)
        self.conv2=nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1)

    def forward(self,x):
        y=F.relu(self.conv1(x))
        y=F.selu(self.conv2(y))

        return F.relu(x+y)

class basic_block2(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(basic_block2,self).__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=2)
        self.conv2=nn.Conv2d(in_channels,out_channels,kernel_size=2,stride=2,padding=1)
        self.conv3=nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=2,padding=1)

    def forward(self,x):
        z=self.conv1(x)
        y=F.relu(self.conv2(x))
        y=self.conv3(y)

        return F.relu(y+z)

class resnet_test(nn.Module):
    def __init__(self):
        super(resnet_test,self).__init__()
        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3)
        self.maxp1=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.resnet1=basic_block(64)
        self.resnet2 = basic_block(64)
        self.resnet3=basic_block2(64,128)
        self.resnet4 = basic_block(128)
        self.resnet5 = basic_block2(128,256)
        self.resnet4 = basic_block(256)
        self.resnet5 = basic_block2(256, 512)
        self.resnet4 = basic_block(512)
        self.avgp1=nn.AvgPool2d(7)
        self.fullc=nn.Linear(512,1000)

    def forward(self,x):
        in_size=x.size(0)
        x = self.resn1(x)
        x = self.resn2(x)
        x = self.resn3(x)
        x = self.resn4(x)
        x = self.resn5(x)
        x = self.resn6(x)
        x = self.resn7(x)
        x = self.resn8(x)
        x = self.avgp1(F.relu(x))
        x = x.view(in_size, -1)
        x = self.fullc(x)

        return F.softmax(x, dim=1)
