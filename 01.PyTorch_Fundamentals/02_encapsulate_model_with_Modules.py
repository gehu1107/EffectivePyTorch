#!/usr/bin/env python3
# -*- coding:utf8 -*-
""" 使用torch.nn.Module模块封装模型，主要是模型参数和模型操作
"""
import torch
import numpy as np

print(torch.__version__)

# 模块就是一个包含模型参数和模型操作的容器
# 线性模型：y = ax + b
# 一般必须有__init__和forward两个成员函数
class MyLinear(torch.nn.Module):
    def __init__(self):
        """模型必须的成员，初始化模型参数及结构"""
        super().__init__()
        #torch.nn.Parameter是Tensor的子类，特点是：requires_grad=True；默认把参数加入到Mudole.parameters中
        self.a = torch.nn.Parameter(torch.rand(1))
        self.b = torch.nn.Parameter(torch.rand(1))

    def forward(self, x):
        """模型必须的成员，表达模型的前向计算方式，后向计算方式也蕴含其中，autograd自动完成"""
        yhat = self.a * x + self.b
        return yhat

x = torch.arange(100, dtype=torch.float32)
#使用模型的方式：实例化一个模型，然后像函数调用一样调用，传参和forward函数对应
net = MyLinear()
y = net(x)

#可以使用parameters成员函数访问所有模型参数
#print(type(net.parameters())) #迭代器
for p in net.parameters():
    print(p)


## 拟合函数 y = 5x^2 + 3 + noise
x = torch.arange(100, dtype=torch.float32) / 100
y = 5 * x + 3 + torch.rand(100) * 0.3
# 定义损失函数和优化器
#功能和torch.nn.functional.mse_loss()一样：区别是mse_loss是torch.nn.functional里面的是函数，没有自己的参数；MSELoss是torch.nn中的一个模块，继承自torch.nn.Module，可以有自己的参数，torch.nn.Module中很多都是自己管理参数的同时调用functional的函数实现具体逻辑；
criterion = torch.nn.MSELoss() #torch.nn.functional.mse_loss()

def train(net, criterion, optimizer, x, y):
    for i in range(10000):
        #前向
        yhat = net(x)
        #loss
        loss = criterion(yhat, y)
        #计算梯度
        net.zero_grad()
        loss.backward()
        #更新梯度
        optimizer.step()
        print(f"round {i}: {loss}")
        if loss < 0.01:
            break
#net = MyLinear()
#optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
#train(net, criterion, optimizer, x, y)
#print(net.a, net.b)

# torch.nn中有很多预定义的模块，上面的就可以用预定义的torch.nn.Linear模块实现
class MyLinear_new(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 使用预定义的线性模块，参数在Linear自己定义好的，只需要指定输入输出的维度
        self.linear = torch.nn.Linear(1,1)

    def forward(self, x):
        # x:n维向量，要变成矩阵，因为torch.nn.Linear处理的输入是批量的矩阵
        yhat = self.linear(x.unsqueeze(1))
        # yhat返回一个n维向量
        return yhat.squeeze(1)

#net = MyLinear_new()
net = MyLinear()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
train(net, criterion, optimizer, x, y)
for p in net.parameters():
    print(p)

# torch预定义的模块有很多，最常用的容器模块是torch.nn.Sequential，其常用于串联多个模块或者网络层，自动按顺序调用；例如实现一个两个Linear层组的网络
model = torch.nn.Sequential(
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 10),
        )
# 调用model(x)时：从上往下依次调用序列中的模块，然后把上一层的输出作为下一层的输入
# 除了Sequential之外，还有ModuleList, ModuleDict类


# pytorch在大的tensors上有很好的优化，所以尽量用批量操作，如果不好手动批量，则可以考虑TorchScipt，pytorch会使用jit自动优化TorchScript代码
# 下面以batch_gather为例, output[i] = input[i, index[i]]
def batch_gather(tensor, indices):
    output = []
    for i in range(tensor.size(0)):
        output += [tensor[i][indices[i]]]
    return torch.stack(output)

#TorchScript版本
@torch.jit.script
def batch_gather_jit(tensor, indices):
    output = []
    for i in range(tensor.size(0)):
        output += [tensor[i][indices[i]]]
    return torch.stack(output)
#jit版本要比非jit版快10%左右

# 但是，最好的还是手动分批量处理，一个向量化的处理要快100倍！
def batch_gather_vec(tensor, indices):
    shape = list(tensor.shape)
    #前两列合一块
    flat_first = torch.reshape(tensor, [shape[0] * shape[1]] + shape[2:])
    #按块求后面的偏移
    offset = torch.reshape(torch.arange(shape[0]) * shape[1], [shape[0]] + [1] * (len(indices.shape) - 1))
    output = flat_first[indices.shape + offset]
    return output
