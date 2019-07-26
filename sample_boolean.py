import numpy as np
from collections import Counter
from itertools import product
from collections import OrderedDict as OD
import torch
from torch import nn
from torch.autograd import Variable


BIT = [-1, 1]

def init_linear(lin, s_b, s_w):
    '''
    Randomly samples the weights and biases of a linear layer.
    $$W_{ij} \sim N(0, s_w^2/\sqrt{fan_in}), b_i \sim N(0, s_b^2)$$
    
    Inputs:
        lin: an `nn.Linear` instance
        s_w: the standard deviation of the weights is `s_w`/sqrt(fan_in)
        s_b: the standard deviation of the biases is `s_b`
    Outputs:
        None.
        We modify `lin` in-place.
    '''
    for p in lin.parameters():
        if p.dim() == 1: # bias
            torch.randn(p.size(), out=p.data).mul_(s_b)
        if p.dim() == 2: # weight
            torch.randn(p.size(), out=p.data).mul_(
                s_w / np.sqrt(p.size()[1]))
class MyNet(nn.Module):
    def __init__(self, nonlin, widthspec):
        '''
        Simple multilayer perceptron given an architectural specfication.
        Inputs:
            `nonlin`: the nonlinearity, as a torch module
            `widthspec`: a specification of the architecture.
                It should be a list of widths, starting from the input layer.
        '''
        super(MyNet, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(widthspec)-1):
            self.layers.append(nn.Linear(widthspec[i], widthspec[i+1]))
        self.nonlins = nn.ModuleList()
        for i in range(len(self.layers)-1):
            self.nonlins.append(nonlin())
    @property
    def inputdim(self):
        return self.layers[0].in_features
    def forward(self, x):
        for (layer, nonlin) in zip(self.layers, self.nonlins):
            x = nonlin(layer(x))
        return self.layers[-1](x)
    def randomize(self, b, w):
        for l in self.layers:
            init_linear(l, b, w)
class Erf(nn.Module):
    def forward(self, x):
        return x.erf()
    
def f2str(f, inputd=None):
    '''
    Given a scalar function on the boolean cube, threshold it to
    a boolean function, and then converting that to a binary string.
    '''
    if inputd is None:
        inputd = f.inputdim
    inputbatch = Variable(torch.Tensor(list(product(BIT, repeat=inputd))))
    outputs = f(inputbatch) > 0
    return ''.join(map(lambda x: str(x), outputs.data.numpy().reshape(-1)))

def sample_boolean_fun(f, vw, vb, n, outformat='list'):
    Cs = Counter()
    for _ in range(n):
        f.randomize(np.sqrt(vb), np.sqrt(vw))
        Cs[f2str(f)] += 1
    if outformat == 'counter':
        return Cs
    elif outformat == 'list':
        return OD(Cs.most_common()).values()
    else:
        raise NotImplementedError('unknown outformat')
