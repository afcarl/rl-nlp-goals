import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(model):
    '''
    Code taken form https://github.com/ikostrikov/pytorch-a3c/blob/master/model.py
    :param model: model the inizialize
    :return: 
    '''
    classname = model.__class__.__name__
    if classname.find('FullyConnected') != -1:
        weight_shape = list(model.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        model.weight.data.uniform_(-w_bound, w_bound)
        model.bias.data.fill_(0)

class FullyConnected(nn.Module):
    def __init__(self, in_features, out_features, activation_fn=lambda x: x, bias=True, dtype=torch.FloatTensor):
        super(FullyConnected, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(dtype(out_features, in_features))
        if bias:
            self.bias = Parameter(dtype(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.activation_fn = activation_fn

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return self.activation_fn(F.linear(input, self.weight, self.bias))

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



def softmax(input, axis=-1):
    input_size = input.size()

    trans_input = input.transpose(axis, len(input_size) - 1)
    trans_size = trans_input.size()

    input_2d = trans_input.contiguous().view(-1, trans_size[-1])

    soft_max_2d = F.softmax(input_2d)

    soft_max_nd = soft_max_2d.view(*trans_size)
    return soft_max_nd.transpose(axis, len(input_size ) -1)