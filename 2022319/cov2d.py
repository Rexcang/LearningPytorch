#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/3/31 19:53
# @Author  : Cheng liwei
# @FileName: cov2d.py
# @Software: PyCharm

import torch
import torch.nn as nn

layer=nn.Conv2d(1,3,kernel_size=3,stride=1,padding=0)
x=torch.randn(1,1,28,28)
out=layer.forward(x)

layer=nn.Conv2d(1,3,kernel_size=3,stride=1,padding=1)
out=layer.forward(x)
layer=nn.Conv2d(1,3,kernel_size=3,stride=2,padding=1)
out=layer.forward(x)
out=layer(x)
layer.weight
layer.weight.shape