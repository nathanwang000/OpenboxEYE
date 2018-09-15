from lib.data import Mimic2
from lib.parallel_run import map_parallel
import numpy as np
from lib.model import MLP
from lib.train import Trainer, prepareData
from lib.openbox import open_box
from torch.utils.data import Dataset, DataLoader, TensorDataset
from lib.regularization import eye_loss, wridge, wlasso, lasso, enet, owl, ridge, no_reg, r4rr
from sklearn.metrics import accuracy_score
from lib.utility import get_y_yhat, model_auc, modelAP, sweepS1, modelSparsity, bootstrap
import torch, os
from scipy.stats import ttest_rel
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import sys
import argparse

def parser():
    parser = argparse.ArgumentParser(description='Use EYE to correct network mistakes given by Openbox')
    parser.add_argument('function', type=str, metavar='f', nargs=1,
                        help='function to evaluate')
    parser.add_argument('regs', type=str, metavar='regs', nargs='?', default='None',
                        help='regularization for function (optional)')
    parser.add_argument('--batch_size', type=int, default=4000, metavar='N',
                        help='input batch size for training (default: 4000)')
    parser.add_argument('--n_cpus', type=int, default=None, metavar='n_cpu',
                        help='number of cpus for training (default: None)')
    parser.add_argument('--n_bootstrap', type=int, default=30, metavar='n_bootstrap',
                        help='number of bootstrap samples for validation (default: 30)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--dup', type=int, default=0, metavar='dup',
                        help='duplicate features # times (default: 0)')
    parser.add_argument('--noise', type=float, default=0.01, metavar='noise',
                        help='noise level for duplicate (default: 0.01)')
    parser.add_argument('--threshold', type=float, default=0.9, metavar='threshold',
                        help='noise level for duplicate (default: 0.9)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log_interval', type=int, default=1, metavar='log_interval',
                        help='how many batches to wait before logging training status')
    # parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
    #                     help='SGD momentum (default: 0.5)')
    # parser.add_argument('--no-cuda', action='store_true', default=False,
    #                     help='disables CUDA training')
    args = parser.parse_args()
    return args

def trainData(name, data, regularization=eye_loss, alpha=0.01, n_epochs=300, 
              learning_rate=1e-3, batch_size=4000, log_interval=1, test=False):
    '''
    return validation auc, average precision, score1
    if test is true, combine train and val and report on test performance
    '''
    m = data

    if test:
        name = 'test' + name
        xtrain = np.vstack([m.xtrain, m.xval])
        xval = m.xte
        ytrain = np.hstack([m.ytrain, m.yval])
        yval = m.yte
    else:
        xtrain = m.xtrain
        xval = m.xval
        ytrain = m.ytrain
        yval = m.yval

    # note: for cross validation, just split data into n fold and
    # choose appropriate train_data and valdata from those folds
    # not doing here for simplicity
    d = m.r.size(0)
    train_data = TensorDataset(*map(lambda x: x.data, prepareData(xtrain, ytrain)))
    data = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    valdata = TensorDataset(*map(lambda x: x.data, prepareData(xval, yval)))
    valdata = DataLoader(valdata, batch_size=batch_size, shuffle=True)

    n_output = 2 # binary classification task
    model = MLP([d, 30, 10, n_output]) 

    t = Trainer(model, lr=learning_rate, risk_factors=m.r, alpha=alpha,
                regularization=regularization,
                name=name)
    losses, vallosses = t.fit(data, n_epochs=n_epochs,
                              print_every=log_interval,
                              valdata=valdata)

    # load model with lowest validation loss
    model = torch.load('models/%s.pt' % name)
    
    # report statistics: 
    val_auc = model_auc(model, valdata)
    ap = modelAP(model, valdata, m.r.data.numpy())
    t, s1 = sweepS1(model, valdata)    
    sp = modelSparsity(model, valdata)
    joblib.dump((val_auc, ap, s1, sp), 'models/' + name + '.pkl')    
    return val_auc, ap, s1, sp

class ParamSearch:
    def __init__(self, data, p): # p is param parser
        self.tasks = []
        self.hyperparams = []
        self.n_cpus = p.n_cpus
        self.data = data
        valdata = TensorDataset(*map(lambda x: x.data,
                                     prepareData(data.xval, data.yval)))
        self.valdata = DataLoader(valdata, batch_size=p.batch_size, shuffle=True)
        self.p = p
        
    def add_param(self, name, reg, alpha):
        if not os.path.exists('models/' + name + '.pkl'):        
            self.tasks.append((name, self.data, reg, alpha, p.epochs,
                               p.lr, p.batch_size, p.log_interval))
        self.hyperparams.append((name, reg, alpha))

    def run(self, n_bootstrap=100):
        #map_parallel(trainData, self.tasks, self.n_cpus)        
        for task in self.tasks:
            trainData(*task)
        
        # select a model to run: split on auc and sparsity
        aucs = []
        models = []
        sparsities = []
        for name, reg, alpha in self.hyperparams:
            # load the model        
            model = torch.load('models/' + name + '.pt')
            sp = modelSparsity(model, self.valdata) 
            models.append(model)
            sparsities.append(sp)

        for _ in range(n_bootstrap):
            test = bootstrap(self.valdata)
            local_aucs = []
            for model in models:
                # bootstrap for CI on auc
                local_aucs.append(model_auc(model, test))
            aucs.append(local_aucs)
        aucs = np.array(aucs)

        # only keep those with high auc
        b = np.argmax(aucs.mean(0))
        discardset = set([])
        for a in range(len(models)):
            diffs = ((aucs[:,a] - aucs[:,b]) >= 0).astype(np.int)
            if diffs.sum() / diffs.shape[0] <= 0.05:
                discardset.add(a)

        # choose the one with largest sparsity
        chosen, sp = max(filter(lambda x: x[0] not in discardset,
                                enumerate(sparsities)),
                         key=lambda x: x[1])

        # retrian the chosen model
        name, reg, alpha = self.hyperparams[chosen]
        print('name', name)        
        trainData(name, self.data, reg, alpha, p.epochs, p.lr, p.batch_size,
                  p.log_interval, test=True)

def random_risk_exp(p): # p is argparser
    m = Mimic2(mode='total', random_risk=True, seed=p.seed)
    ps = ParamSearch(m, p)

    reg = eye_loss    
    alphas = [0.1, 0.01, 0.001, 0.0001, 0.00001]

    for alpha in alphas:
        name = 'random_risk_eye' + '^' + str(alpha)
        ps.add_param(name, reg, alpha)

    ps.run(p.n_bootstrap)

def reg_exp(p):
    m = Mimic2(mode='total', seed=p.seed)
    ps = ParamSearch(m, p)
    
    alphas = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    for reg in p.regs:
        for alpha in alphas:
            name = reg.__name__ + '^' + str(alpha)
            ps.add_param(name, reg, alpha)

    ps.run(p.n_bootstrap)

def duplicate_exp(p):
    m = Mimic2(mode='total', duplicate=p.dup, noise=p.noise, seed=p.seed)
    ps = ParamSearch(m, p)
    
    alphas = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    for reg in p.regs:
        for alpha in alphas:
            name = reg.__name__ + '_dup' + str(p.dup) + '_' + str(p.noise)\
                   + '^' + str(alpha)
            ps.add_param(name, reg, alpha)

    ps.run(p.n_bootstrap)

def expert_feature_only_exp(p):    
    m = Mimic2(mode='total', expert_feature_only=True, seed=p.seed)
    ps = ParamSearch(m, p)

    reg = ridge
    alphas = [0.1, 0.01, 0.001, 0.0001, 0.00001]

    for alpha in alphas:
        name = 'expert_only_ridge' + '^' + str(alpha)
        ps.add_param(name, reg, alpha)

    ps.run(p.n_bootstrap)

def two_stage_exp(p):
    '''
    remove features by setting a threshold on correlation, 
    then apply l2 regularization on the remaining features
    '''
    m = Mimic2(mode='total', two_stage=True, threshold=p.threshold, seed=p.seed)
    ps = ParamSearch(m, p)

    reg = ridge
    alphas = [0.1, 0.01, 0.001, 0.0001, 0.00001]

    for alpha in alphas:
        name = 'two_stage_ridge_' + str(p.threshold) + '^' + str(alpha)
        ps.add_param(name, reg, alpha)

    ps.run(p.n_bootstrap)

def dummy(p):
    '''
    remove features by setting a threshold on correlation, 
    then apply l2 regularization on the remaining features
    '''
    print('dummy run')
    print(p.regs, p.n_cpus, p.n_bootstrap, p.batch_size, p.seed)
    
#####################################################
def wridge1_5(*args, **kwargs):
    return wridge(*args, **kwargs, w=1.5)

def wridge3(*args, **kwargs):
    return wridge(*args, **kwargs, w=3)

def wlasso1_5(*args, **kwargs):
    return wlasso(*args, **kwargs, w=1.5)

def wlasso3(*args, **kwargs):
    return wlasso(*args, **kwargs, w=3)

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('please specify your function and argument to run')
    else:
        p = parser()
        print(p.regs)
        p.regs = eval(p.regs)
        print('running function', p.function[0])
        f = eval(p.function[0])
        f(p)

