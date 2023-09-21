



import torch
import torch.nn.functional as F
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(Linear, self).__init__()

        self.weight = torch.randn((in_dim,out_dim))
        self.bias = torch.randn((out_dim))

    def forward(self,x):
        # x[*,in_dim]
        # weight[in_dim,out_dim]
        # x = torch.mm(x,self.weight) + self.bias
        # 矩阵乘法
        x = x @ self.weight + self.bias
        return x

class Conv2d(nn.Module):
    def __init__(self,in_dim,out_dim,kernel_size,stride,padding):
        super(Conv2d, self).__init__()
        self.weight = torch.randn((out_dim,in_dim,kernel_size,kernel_size))
        self.bias = torch.randn((out_dim))
        self.stride = stride
        self.padding = padding
    def forward(self,x):
        # x = [B,C,H,W]
        # weight = [out_dim,in_dim,kernel_size,kernel_size]
        # bias = [out_dim]
        B,C,H,W = x.shape
        for i in range(B):
            pass
        return x