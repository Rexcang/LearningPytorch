#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/4/5 20:30
# @Author  : Cheng liwei
# @FileName: lenet5.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as  F


class Lennet5(nn.Module):

    ##
    ##for cifar
    ##

    def __init__(self):
        super(Lennet5,self).__init__()

        self.conv_unit=nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0),
            nn.Conv2d(6,16,kernel_size=5,stride=1,padding=0),
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0)
        )

        ##flatten
        #fc unit
        self.fc_unit=nn.Sequential(
            nn.Linear(16*5*5,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
        )



        #[b,16,5,5]
        # tmp=torch.randn(2,3,32,32)
        # out=self.conv_unit(tmp)
        # print('conv_out',out.shape)

    def forward(self,x):

        #[b,3,32,32]

        batchsz=x.size(0)
        x=self.conv_unit(x)
        #[b,16,5,5]==>[b,16*5*5]
        x=x.view(batchsz,16*5*5)

        #[b,16*5*5]<-->[b,10]
        logits=self.fc_unit(x)

        #[b,10]  在10dim上做一个分类，所以 dim=1
      #  pred=F.softmax(logits,dim=1)
      #在使用crossentropyloss时，包含了softmax操作，所以，不需要再使用了
        #loss的计算  use Cross empty loss ,对于分类问题，使用交叉熵更加合适
        #loss=

        return logits

def main():

    net=Lennet5()

    tmp=torch.randn(2,3,32,32)
    out=net(tmp)
    print('conv_out',out.shape)

if __name__ == '__main__':
    main()
