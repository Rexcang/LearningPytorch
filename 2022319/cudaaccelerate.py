#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/3/19 21:21
# @Author  : Cheng liwei
# @FileName: cudaaccelerate.py
# @Software: PyCharm

import torch
import time

from torch import autograd

print(torch.__version__)
print(torch.cuda.is_available())

a=torch.randn(10000,1000)           ##torch.randn作用是返回一个张量，前面的变量定义了size,张量的值是从0-1随机均匀的提取
b=torch.randn(1000,2000)

t0=time.time()                      #time.time()是提取时间戳
c=torch.matmul(a,b)                 #torch.matmul()是张量的乘法 出入多维时，把多出的一维batch出来，其余做矩阵乘法
t1=time.time()
print(a.device,t1-t0,c.norm(2))     ###norm即求解p范数
                                    ###torch.device() 分配到对应的设备上

#cpu和cuda之间的转换使用cuda来实现
device=torch.device('cuda')         #torch.device 更改部署的设备
print(device)
a=a.to(device)
b=b.to(device)

t0=time.time()
c=torch.matmul(a,b)
t2=time.time()
print(a.device,t2-t0,c.norm(2))


t0=time.time()
c=torch.matmul(a,b)
t3=time.time()
print(a.device,t3-t0,c.norm(2))

#####深度学习就是一个关于梯度的问题 求解 y=a^2*x+bc+c 求x=1的三个导数
##
###pytorch 变量定义 赋值 以及是否自动求导

##四个变量分别赋值为 1 1 2 3 requires_grad=True表示自动求导
x=torch.tensor(1)
a=torch.tensor(1.,requires_grad=True)
b=torch.tensor(2.,requires_grad=True)
c=torch.tensor(3.,requires_grad=True)

##**表示指数运算
y=a**2*x+b*x+c

print('before:',a.grad,b.grad,c.grad)
# gradns=autograd.grad(y,a)
# print(gradns)
grads=autograd.grad(y,[a,b,c])
print('after:',grads[0],grads[1],grads[2])

##autograd.grad(output,inmput)

##常用的网络层
# nn.Linear
# nn.Conv23
# nn.LSTM
#
# nn.ReLU
# nn.Sigmoid
#
# nn.Softmax
# nn.CrossEntropyLoss
# nn.MSE

