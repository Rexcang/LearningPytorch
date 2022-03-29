#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/3/28 21:17
# @Author  : Cheng liwei
# @FileName: TEST.py
# @Software: PyCharm
import torch

from torch import nn
from torch.nn import functional as F
from torch import optim

x=torch.randn(1,784)
w=torch.randn(10,784)

logits=x@w.t()
pred=F.softmax(logits,dim=1)
pred_log=torch.log(pred)
a=F.cross_entropy(logits,torch.tensor([4]))   #包括了计算softmax和交叉熵
print(a)