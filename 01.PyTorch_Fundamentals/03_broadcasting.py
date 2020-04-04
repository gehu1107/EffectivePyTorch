#!/usr/bin/env python3
# -*- coding:utf8 -*-
""" pytorch的广播机制的优缺点
一般的tensor操作要明确指定操作的维度和以及使用torch.unsqueeze和torch.squeeze函数
"""
import torch
import numpy as np

print(torch.__version__)

# 类似numpy的广播机制，主要用在加法和乘法上
a = torch.tensor([[1., 2.], [3., 4.]])
print(a.shape)
b = torch.tensor([[1.], [2.]])
print(b.shape)
# torch.repeat和numpy.tile很像, b.shape[0]放大一倍，b.shape[1]放大2倍
# b：2*1 变成(2*1)*(1*2) = 2*2
#c = a + b.repeat(1, 2)
# 广播机制等价与上面
c = a + b
print(c.shape)
print(c)


# 神经网络中经常会把输入通过tile和cat合并
a = torch.rand([5, 3, 5])
b = torch.rand([5, 1, 6])
linear = torch.nn.Linear(11, 10)
# concat a and b and apply nonlinearity
tiled_b = b.repeat([1, 3, 1])
c = torch.cat([a, tiled_b], 2)
d = torch.nn.functional.relu(linear(c))
print(d.shape)  # torch.Size([5, 3, 10])

# 上述可以直接利用torch的广播机制更优雅的实现
a = torch.rand([5, 3, 5])
b = torch.rand([5, 1, 6])
linear1 = torch.nn.Linear(5, 10)
linear2 = torch.nn.Linear(6, 10)
pa = linear1(a)     #5,3,10
pb = linear2(b)     #5,1,10
pc = pa + pb
d = torch.nn.functional.relu(pc)
print(d.shape)  # torch.Size([5, 3, 10])


# 事实上，上述代码可以应用到任意形状的tensor
class Merge(torch.nn.Module):
    def __init__(self, in_features1, in_features2, out_features, activation=None):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features1, out_features)
        self.linear2 = torch.nn.Linear(in_features2, out_features)
        self.activation = activation

    def forward(self, a, b):
        pa = self.linear1(a)
        pb = self.linear2(b)
        pc = pa + pb
        if self.activation is not None:
            pc = self.activation(pc)
        return pc


# 广播机制的缺点
a = torch.tensor([[1.], [2.]]) #2,1
b = torch.tensor([1., 2.])  #1,2
# a+b: 2,2
c = torch.sum(a + b)
print(c) #12
d = torch.sum(a+b, 0)
print(d) #(5,7)     #就立即知道错误
