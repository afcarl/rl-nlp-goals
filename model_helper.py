import torch
import torch.nn as nn
import torch.nn.functional as F


class FullyConnected(nn.Module):
    def __init__(self, input_dim, output_dim, activation_fn=lambda x: x):
        super(FullyConnected, self).__init__()
        self.layer = nn.Linear(input_dim, output_dim)
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.layer(x)
        return self.activation_fn(x)



def softmax(input, axis=-1):
    input_size = input.size()

    trans_input = input.transpose(axis, len(input_size) - 1)
    trans_size = trans_input.size()

    input_2d = trans_input.contiguous().view(-1, trans_size[-1])

    soft_max_2d = F.softmax(input_2d)

    soft_max_nd = soft_max_2d.view(*trans_size)
    return soft_max_nd.transpose(axis, len(input_size ) -1)