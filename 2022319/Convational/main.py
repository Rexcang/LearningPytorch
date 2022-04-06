#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/4/4 21:46
# @Author  : Cheng liwei
# @FileName: main.py.py
# @Software: PyCharm



import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch import nn,optim

from lenet5 import Lennet5
def main():
    batchsz=32

    cifar_train=datasets.CIFAR10('cifar',True,transform=transforms.Compose([
        transforms.Resize([32,32]),
        transforms.ToTensor()
    ]),download=True)

    cifar_train=DataLoader(cifar_train,batch_size=batchsz,shuffle=True)

    cifar_test= datasets.CIFAR10('cifar', False, transform=transforms.Compose([
        transforms.Resize([32,32]),
        transforms.ToTensor()
    ]), download=True)

    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)

    x,label=iter(cifar_train).next()
    print('x:',x.shape,'label:',label.shape)

    device = torch.device('cuda')
    model=Lennet5().to(device)
    criteon=nn.CrossEntropyLoss().to(device)
    optimizer=optim.Adam(model.parameters(),lr=1e-3)



    print(model)

    for epoch in range(1000):

        model.train()
        for batchidx,(x,label) in enumerate(cifar_train):
            #x  [b,3,32,32]
            #label b
            x,label=x.to(device),label.to(device)

            logists=model(x)
            #logists [b,10]
            #label b
            #$loss tensor scalar
            loss=criteon(logists,label)

            #backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

         #
        print(epoch,loss.item())


        #test

        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_num=0
            acc=0
            for x,label in cifar_test:
                x,label=x.to(device), label.to(device)

                logists=model(x)

                pred=logists.argmax(dim=1)
                total_correct+=torch.eq(pred,label).float().sum()
                total_num+=x.size(0)

            acc= total_correct / total_num
            print(epoch,acc)

def main():



if __name__ == '__main__':
    main()


