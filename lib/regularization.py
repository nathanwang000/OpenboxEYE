import numpy as np
import sys
import torch
from torch import nn
from torch.autograd import Variable

def no_reg(loss, alpha=0, r=None): # placeholder with no regularization

    def ret(yhat, y, theta):
        return loss(yhat, y)

    return ret
    
def ridge(loss, alpha, r=None):

    def reg(x):
        return 0.5 * (x).dot(x)

    def ret(yhat, y, Ws):
        res = 0
        for W in Ws:
            res += reg(W[1] - W[0])
        return loss(yhat, y) + res / len(Ws) * alpha
    
    return ret

def wridge(loss, alpha, r, w=2):

    def reg(x):
        # nonlocal r # default to all unknown
        # r = r or Variable(torch.zeros(x.numel()), requires_grad=False)
        weight = w - (w-1) * r
        return 0.5 * (x * weight).dot(x * weight)

    def ret(yhat, y, Ws):
        res = 0
        for W in Ws:
            res += reg(W[1] - W[0])
        return loss(yhat, y) + res / len(Ws) * alpha
    
    return ret

def lasso(loss, alpha, r=None):
    def reg(x):
        return torch.abs(x).sum()

    def ret(yhat, y, Ws):
        res = 0
        for W in Ws:
            res += reg(W[1] - W[0])
        return loss(yhat, y) + res / len(Ws) * alpha

    return ret

def wlasso(loss, alpha, r, w=2):

    def reg(x):
        # nonlocal r # default to all unknown
        # r = r or Variable(torch.zeros(x.numel()), requires_grad=False)
        weight = w - (w-1) * r
        return torch.abs(x * weight).sum()

    def ret(yhat, y, Ws):
        res = 0
        for W in Ws:
            res += reg(W[1] - W[0])
        return loss(yhat, y) + res / len(Ws) * alpha
    
    return ret

def owl(loss, alpha, r=None):

    def reg(x):
        # the infinity norm formulation    
        weight = Variable(torch.zeros(x.numel()))
        weight.data[-1] = 1 # because order is sorted ascending
        
        order = torch.from_numpy(np.argsort(x.abs().data.numpy())).long()
        return (weight * x.abs()[order]).sum()

    def ret(yhat, y, Ws):
        res = 0
        for W in Ws:
            res += reg(W[1] - W[0])
        return loss(yhat, y) + res / len(Ws) * alpha
    
    return ret

def enet(loss, alpha, r=None, l1_ratio=0.5):

    def reg(x): 
        return l1_ratio * torch.abs(x).sum() + (1-l1_ratio) * 0.5 * x.dot(x)

    def ret(yhat, y, Ws):
        res = 0
        for W in Ws:
            res += reg(W[1] - W[0])
        return loss(yhat, y) + res / len(Ws) * alpha
    
    return ret

def eye_loss(loss, alpha, r): # loss is the data loss

    def reg(x):
        # nonlocal r # default to all unknown
        # r = r or Variable(torch.zeros(x.numel()), requires_grad=False)
        l1 = torch.abs(x * (1-r)).sum()
        l2sq = (r * x).dot(r * x)
        return  l1 + torch.sqrt(l1**2 + l2sq)

    def ret(yhat, y, Ws):
        res = 0
        for W in Ws:
            res += reg(W[1] - W[0])
        return loss(yhat, y) + res / len(Ws) * alpha
    
    return ret
