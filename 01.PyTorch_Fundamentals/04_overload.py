#!/usr/bin/env python3
# -*- coding:utf8 -*-
""" pytorch的重载
"""
import torch
import time
import numpy as np

print(torch.__version__)

# 和numpy一样，pytorch也重载了一些Python的操作
# sliciing
x = torch.arange(10)
z = x[2:4]
# z = x.narrow(0, 2, 2)
print(z)

x = torch.rand(500, 10)
start = time.time()
## 循环累加
#z = torch.zeros(10)
#for i in range(500):
#    z += x[i]
## 0.002919 seconds

## 更好的选择是使用torch.unbind函数，把x矩阵拆成tensor序列，
#z = torch.zeros(10)
#for x_i in x.unbind():
#    z += x_i
## 0.003087 seconds.

## 最好的当然还是使用重载的sum函数
#z = x.sum(dim=0)
## 0.000898 seconds.
print("Took %f seconds." % (time.time() - start))


# 其他pytorch常用的重载操作符
z = -x  # z = torch.neg(x)
z = x + y  # z = torch.add(x, y)
z = x - y
z = x * y  # z = torch.mul(x, y)
z = x / y  # z = torch.div(x, y)
z = x // y
z = x % y
z = x ** y  # z = torch.pow(x, y)
z = x @ y  # z = torch.matmul(x, y)
z = x > y
z = x >= y
z = x < y
z = x <= y
z = abs(x)  # z = torch.abs(x)
z = x & y
z = x | y
z = x ^ y  # z = torch.logical_xor(x, y)
z = ~x  # z = torch.logical_not(x)
z = x == y  # z = torch.eq(x, y)
z = x != y  # z = torch.ne(x, y)

# 另外也可以用自增的，例如，z += x, z **= 2等
# 但是，Python不允许重载"and", "or", "not"等关键操作
