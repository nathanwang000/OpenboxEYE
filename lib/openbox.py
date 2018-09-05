from torch import nn
import torch
from torch.autograd import Variable
import numpy as np
import tqdm
from collections import defaultdict
from lib.utility import to_var

'''
todo: make a recursive definition of open_box to account for residue connection

current implementation can be made more efficient by memoization of bottom up search through the network
'''

# assumes has an attribute called classifier with sequential layers
def open_box(model, x): 
    # forward pass to determine configuration
    # assume x is flat with no batch dimension
    assert len(x.shape) == 1, "assume no batch dimension in input"
    d = x.shape[0]
    C = []
    
    # get W and b
    W = to_var(torch.eye(d))
    b = to_var(torch.zeros(d))
    z = x
    
    for i, c in enumerate(model.classifier):
        if type(c) == torch.nn.modules.linear.Linear:
            W = torch.mm(c.weight, W)
            b = c.bias + torch.mv(c.weight, b)
        elif type(c) == torch.nn.modules.ReLU:
            C.extend(list((z > 0).int().data.numpy())) # configuration
            r = (z > 0).float() # the slope
            t = torch.zeros_like(z) # the bias
            W = torch.mm(torch.diag(r), W)
            b = t + torch.mv(torch.diag(r), b)
        elif type(c) == torch.nn.modules.LeakyReLU:
            C.extend(list((z > 0).int().data.numpy())) # configuration
            r = (z > 0).float() # the slope
            r[r==0] = c.negative_slope
            t = torch.zeros_like(z) # the bias
            W = torch.mm(torch.diag(r), W)
            b = t + torch.mv(torch.diag(r), b)            
        else:
            raise Exception('unknown layer')
            
        z = c(z) # forward pass
    
    C = ''.join(map(str, C))
    return W, b, C

def open_box_batch(model, x):
    # batch mode openbox
    assert len(x.shape) == 2, "assume have batch dimension in input"
    batch_size = x.shape[0]

    Ws, bs, Cs = [], [], []
    for i in range(batch_size):
        W, b, C = open_box(model, x[i])
        Ws.append(W)
        bs.append(b)
        Cs.append(C)

    return Ws, bs, Cs

def count_config(net, loader):
    counter = {}
    for x, y in tqdm.tqdm(loader):
        for i, im in enumerate(x):
            target = y[i].item()
            W, b, C = open_box(net, im.view(-1))
            if not counter.get(C, None):
                counter[C] = defaultdict(int)
            counter[C][target] = counter[C].get(target, 0) + 1
    return counter

# find an instance with config
def find_x(config, net, loader):
    for x, y in loader:
        for i, im in enumerate(x):
            W, b, C = open_box(net, im.view(-1))
            if C == config:
                return im

def find_all_x(config, net, loader):
    cases = []
    for x, y in loader:
        for i, im in enumerate(x):
            W, b, C = open_box(net, im.view(-1))
            if C == config:
                cases.append(im.unsqueeze(0))
    return torch.cat(cases, 0)
            
