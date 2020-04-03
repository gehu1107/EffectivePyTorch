#!/usr/bin/env python3
# -*- coding:utf8 -*-

"""拟合f(x)
"""
import torch
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

#参数
w = torch.autograd.Variable(torch.rand(4,1), requires_grad=True)

#要拟合的函数为f(x) = 5 * x^3 + x - 10
def f(x):
    return 5*(x**3) + x - 10

#假设 fhat(x) = a*x^3 + b*x^2 + c*x + d; a = 5, b = 0, c = 1, d = -10
def fhat(x):
    return w[0] * (x**3) + w[1] * (x**2) + w[2] * x + w[3]

def gen_data(batch=100):
    """生产样本，加入随机噪声，每次100个样本
    """
    x = torch.rand(batch) * 20 + 10
    y = f(x) #+ torch.rand(batch)
    return x, y

def calc_loss(yhat, y):
    """计算损失"""
    return torch.nn.functional.mse_loss(yhat, y)


optimizer = torch.optim.Adam([w], lr=0.1)
#optimizer = torch.optim.SGD([w], lr=0.1, momentum=0.9)
#使用生成的数据拟合函数
for i in range(100000):
    #获取数据
    x, y = gen_data(256)
    y = y.to(device=device)
    #前向计算
    yhat = fhat(x).to(device=device)
    #计算损失
    loss = calc_loss(yhat, y)
    #计算梯度
    optimizer.zero_grad()
    loss.backward()
    #更新梯度
    optimizer.step()
    if i % 100 == 0:
        print(f"{loss}")
    if loss < 0.1:
        break

#[5, 0, 1, 10]
print(w.detach().numpy())
