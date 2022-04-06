#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/4/6 20:18
# @Author  : Cheng liwei
# @FileName: resnet.py
# @Software: PyCharm
import torch
from torch import nn
from torch.nn import functional as F


class ResBLK(nn.Module):

    def __init__(self,ch_in,ch_out):

        super(ResBLK,self).__init__()

        self.conv1=nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(ch_out)
        self.conv2=nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn2=nn.BatchNorm2d(ch_out)

        self.extra=nn.Sequential()
        if ch_out != ch_in:
            self.extra=nn.Sequential(
                nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self,x):

        out=F.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))

        #shout cut
        out=self.extra(x)+out

        return out


class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18,self).__init__()

        self.conv1=nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(16)
        )
        #followed 4 blocks

        self.blk1=ResBLK(16,16)
        self.blk2=ResBLK(16,32)
        self.blk3=ResBLK(256,512)
        self.blk4=ResBLK(512,512)

        self.outlayer=nn.Linear(512*1*1,10)

    def forward(self,x):

        x=F.relu(self.conv1(x))

        x=self.blk1(x)
        x=self.blk2(x)
        x=self.blk3(x)
        x=self.blk4(x)

        x=self.outlayer(x)

        return x


def main():
    print('11')
    blk=ResBLK(64,128,stride=2)
    tmp = torch.randn(2, 3, 32, 32)
    out=blk(tmp)
    print(out.shape)


if __name__=='__main__':
    main()