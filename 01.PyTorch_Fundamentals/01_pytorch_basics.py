#!/usr/bin/env python3
# -*- coding:utf8 -*-

import torch
import numpy as np

# Tensor can store a scalar value
a = torch.tensor(3)
print(a)  # tensor(3)
# or an array
b = torch.tensor([1, 2])
print(b)  # tensor([1, 2])
# or a matrix
c = torch.zeros([2, 2])
print(c)  # tensor([[0., 0.], [0., 0.]])
# or any arbitrary dimensional tensor
d = torch.rand([2, 2, 2])
print(d)


## matrix multiple
# @ 代表矩阵乘法 matmul
x = torch.randn([3, 5])
y = torch.randn([5, 4])
z = x @ y
d = torch.matmul(x,y)
# z == d
#两个矩阵相加，减，乘，除等

# tensor -> numpy array, 使用torch.numpy()函数
print(z.numpy())
# numpy array -> tensor, 使用from_numpy()或者直接用torch.tensor()
narr = np.random.normal(size=(3,4))
#下面两个等价
print(torch.from_numpy(narr))
print(torch.tensor(narr))


#automatic differentiation 自动微分，使用torch.autograd.grad(y, x)
x = torch.tensor(1.0, requires_grad=True)
def u(x):
    return x*x
def g(u):
    return -2*u
#g(x) = -2*x*x, dgdx=dgdu * dudx = -2 * 2x = -4
y = g(u(x))
dgdx = torch.autograd.grad(y, x)[0]
print(dgdx) #-4


# curvr fitting 曲线拟合
#初始的参数
#假设目标函数为f(x) = ax^2 + bx + c
w = torch.randn(3,1).requires_grad_(True)
opt = torch.optim.Adam([w], lr=0.1)

def model(x):
    return w[0]*x*x + w[1]*x + w[2]
#f = torch.stack([x * x, x, torch.ones_like(x)], 1)
#yhat = torch.squeeze(f @ w, 1)
    return yhat

def compute_loss(y, yhat):
    loss = torch.nn.functional.mse_loss(yhat, y)
    return loss

def generate_data():
    # y = 5x*x + x - 10
    x = torch.rand(100) * 20 - 10
    y = 5 * x * x + 3
    return x, y

def train_step():
    #数据
    x, y = generate_data()
    #前向计算loss
    yhat = model(x)
    loss = compute_loss(yhat, y)
    #后向计算梯度
    opt.zero_grad()
    loss.backward()
    #更新梯度
    opt.step()

#训练
for _ in range(1000):
    train_step()

print(w.detach().numpy())
