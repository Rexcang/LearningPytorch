#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/3/26 16:44
# @Author  : Cheng liwei
# @FileName: temple.py
# @Software: PyCharm

from torch import nn
from torch.nn import functional as F

from matplotlib import pyplot  as plt
from utils import plot_image,plot_curve,one_hot
from torch import optim

import torchvision

class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()  #对集成自父类的属性进行初始化，意味着子类继承了父类的所有属性和方法

        self.fc1=nn.Linear(28*28,256)
        self.fc2=nn.Linear(256,64)
        self.fc3=nn.Linear(64,10)

    def forward(self,x):
        # x:[b,1,28,28]
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)

        return x

net =Net()

for epoch in range(3):
    for batch_idx,(x,y) in enumerate(train_loader):

        # x:[b,1,28,28],y:[512]
        #[b,1,28,28] =>[b,feature]
        #[b,1,28,28] =>[b,784]
        x=x.view(x,size(0),28*28)
        out=net(x)

        #[b,10]
        y_onehot=one_hot(y)

        #loss = mse(out,_y_onehot)

