#!/usr/bin/env python
# encoding: utf-8
'''
@author: wujiyang
@contact: wujiyang@hust.edu.cn
@file: CosineMarginProduct.py
@time: 2018/12/25 9:13
@desc: additive cosine margin for cosface
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
def cosine_sim(x1, x2, dim=1, eps=1e-8): # batch*512 512*num
    ip = torch.mm(x1, x2.t()) # batch*num
    # ip = torch.mm(x1, x2)
    w1 = torch.norm(x1, 2, dim) # x1 norm
    w2 = torch.norm(x2, 2, dim)# x2 norm
    return ip / torch.ger(w1,w2).clamp(min=eps) #ger叉乘

class MarginCosineProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """

    def __init__(self, in_features, out_features, scale=30.0, m=0.40):
        super(MarginCosineProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = scale
        self.m = m 
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):

        cosine = cosine_sim(input, self.weight) # shape : B X num_cls

        one_hot = torch.zeros_like(cosine) # B X Num_cls
        one_hot.scatter_(1, label.view(-1, 1), 1.0)# batch *num_cls
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = self.s * (cosine - one_hot * self.m)

        return output # B X Num_CLS

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')' 

if __name__ == '__main__':
    pass
