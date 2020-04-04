#!/usr/bin/env python3
# -*- coding:utf8 -*-
""" 数值计算的稳定性
"""
import torch
import numpy as np

print(torch.__version__)


# 数值计算不仅要保证正确，还要保证稳定性
# x*y/y != x, 当x非常小或者非常大时，都会出错
x = np.float32(1)
y = np.float32(1e-50)  # y would be stored as 0.0
print(y) #0.0
y = np.float32(1e39)  # y would be stored as inf
print(y) #inf
z = x * y / y
#RuntimeWarning: invalid value encountered in float_scalars
print(z)  # prints nan


# 在PyTorch中，float32[1e-45, 3.4028235e+38], 正数部分，下面存为0.0，上面存为inf
print(np.nextafter(np.float32(0), np.float32(1)))  #1e-45
print(np.finfo(np.float32).min)  #-3.4028235e+38
print(np.finfo(np.float32).max)  #3.4028235e+38


## 在计算梯度时非常容易上溢或者下溢
# 以softmax为例
def unstable_softmax(logits):
    exp = torch.exp(logits)
    print(exp)
    return exp / torch.sum(exp)

print(unstable_softmax(torch.tensor([100., 0.])).numpy())  #[ nan, 0.]

# exp(x-c) / sum(exp(x-c)) = exp(x)，所以我们可以将x减去其最大值，则x为[-inf, 0], exp(x)为[0.0, 1.0]
def softmax(logits):
    exp = torch.exp(logits - torch.max(logits))
    return exp / torch.sum(exp)

print(softmax(torch.tensor([1000., 0.])).numpy())  # prints [ 1., 0.]

# 以交叉熵为例
def unstable_softmax_cross_entropy(labels, logits):
    logits = torch.log(softmax(logits))
    return -torch.sum(labels * logits)
labels = torch.tensor([0.5, 0.5])
logits = torch.tensor([1000., 0.])
xe = unstable_softmax_cross_entropy(labels, logits)
print(xe.numpy())  # prints inf

# 稳定版
def softmax_cross_entropy(labels, logits, dim=-1):
    scaled_logits = logits - torch.max(logits)
    normalized_logits = scaled_logits - torch.logsumexp(scaled_logits, dim)
    return -torch.sum(labels * normalized_logits)
labels = torch.tensor([0.5, 0.5])
logits = torch.tensor([1000., 0.])
xe = softmax_cross_entropy(labels, logits)
print(xe.numpy())  # prints 500.0

#使用梯度验证下
logits.requires_grad_(True)
xe = softmax_cross_entropy(labels, logits)
g = torch.autograd.grad(xe, logits)[0]
print(g.numpy())  # prints [0.5, -0.5]
