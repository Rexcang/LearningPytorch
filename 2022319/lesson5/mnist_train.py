#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/3/26 15:36
# @Author  : Cheng liwei
# @FileName: mnist_train.py
# @Software: PyCharm

#load
#bulit
#train
#test

import torch

from torch import nn
from torch.nn import functional as F
from torch import optim

import torchvision

from matplotlib import pyplot  as plt
from utils import plot_image,plot_curve,one_hot

batch_size = 512

#
#batch_size 一次处理的图片的数量

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=False)

x,y=next(iter(train_loader))
print(x.shape,y.shape,x.min(),x.max())
plot_image(x,y,'image_sample')

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
        x=self.fc3(x)

        return x

net =Net()
#parameters返回[w1,b1,w2,b2,w3,b3]
optimizer=optim.SGD(net.parameters(),lr=0.01,momentum=0.9)

train_loss=[]

for epoch in range(3):
    for batch_idx,(x,y) in enumerate(train_loader):

        # x:[b,1,28,28],y:[512]
        #[b,1,28,28] =>[b,feature]
        #[b,1,28,28] =>[b,784]
        x=x.view(x.size(0),28*28)
        out=net(x)

        #[b,10]
        y_onehot=one_hot(y)

        #loss = mse(out,_y_onehot)
        loss=F.mse_loss(out,y_onehot)

        optimizer.zero_grad()
        loss.backward()
        #w'=w-lr*grad
        optimizer.step()

        train_loss.append(loss.item())

        if batch_idx %10==0:
            print(epoch,batch_idx,loss.item())

#we get optimal [w1,b1,w2,b2,w3,b3]


plot_curve(train_loss)

total_correct=0

for x,y in test_loader:
    x=x.view(x.size(0),28*28)
    out=net(x)
    #out:[b,10] => pred:[b]
    pred=out.argmax(dim=1)
    correct=pred.eq(y).sum().float().item()
    total_correct+=correct

total_num=len(test_loader.dataset)

acc=total_correct/total_num

print('test acc:',acc)

x,y=next(iter(test_loader))
     out=net(x.view(x.size(0),28*28))
pred=out.argmax(dim=1)
plot_image(x,pred,'test')
