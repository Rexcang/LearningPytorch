#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/3/28 20:13
# @Author  : Cheng liwei
# @FileName: optium.py
# @Software: PyCharm\

import numpy as np
import torch

from matplotlib import pyplot as plt

def himmelblau(x):
    return (x[0]**2 + x[1]-11)**2 + (x[0]+x[1]**2-7)**2

x=np.arange(-6,6,0.1)
y=np.arange(-6,6,0.1)
print('x,y range:', x.shape,y.shape)
X,Y=np.meshgrid(x,y)                             #产生一个以向量x为行，向量y为列的矩阵
print('X,Y maps:',X.shape,Y.shape)
Z=himmelblau([X,Y])

fig=plt.figure('himmelblau')
ax=fig.gca(projection='3d')
ax.plot_surface(X,Y,Z)
ax.view_init(60,-30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()


x=torch.tensor([3.,2.],requires_grad=True)
optimizer=torch.optim.Adam([x],lr=1e-3)

for step in range(20000):
    pred=himmelblau(x)
    optimizer.zero_grad()
    pred.backward()
    optimizer.step()

    if step%2000 ==0:
        print('step {}:x={},f(x)={}'.format(step,x.tolist(),pred.item()))