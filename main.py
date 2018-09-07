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
    # todo
    parser = argparse.ArgumentParser(description='Use EYE to correct network mistakes given by Openbox')
    # parser.add_argument('--batch-size', type=int, default=64, metavar='N',
    #                     help='input batch size for training (default: 64)')
    # parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
    #                     help='input batch size for testing (default: 1000)')
    # parser.add_argument('--epochs', type=int, default=10, metavar='N',
    #                     help='number of epochs to train (default: 10)')
    # parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
    #                     help='learning rate (default: 0.01)')
    # parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
    #                     help='SGD momentum (default: 0.5)')
    # parser.add_argument('--no-cuda', action='store_true', default=False,
    #                     help='disables CUDA training')
    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')
    # parser.add_argument('--log-interval', type=int, default=10, metavar='N',
    #                     help='how many batches to wait before logging training status')
    args = parser.parse_args()
    return args

def trainData(name, data, regularization=eye_loss, alpha=0.01, n_epochs=300, 
              learning_rate=1e-3, batch_size=4000, r=None, test=False):
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
    valdata = DataLoader(valdata, batch_size=4000, shuffle=True)

    n_output = 2 # binary classification task
    model = MLP([d, 8, n_output]) 

    t = Trainer(model, lr=learning_rate, risk_factors=m.r, alpha=alpha,
                regularization=regularization,
                name=name)
    losses, vallosses = t.fit(data, n_epochs=n_epochs, print_every=1, valdata=valdata)

    # report statistics: 
    val_auc = model_auc(model, valdata)
    ap = modelAP(model, valdata, m.r.data.numpy())
    t, s1 = sweepS1(model, valdata)    
    sp = modelSparsity(model, valdata)
    joblib.dump((val_auc, ap, s1, sp), 'models/' + name + '.pkl')    
    return val_auc, ap, s1, sp

class ParamSearch:
    def __init__(self, data, n_cpus=None):
        self.tasks = []
        self.hyperparams = []
        self.n_cpus = n_cpus
        self.data = data
        valdata = TensorDataset(*map(lambda x: x.data,
                                     prepareData(data.xval, data.yval)))
        self.valdata = DataLoader(valdata, batch_size=4000, shuffle=True)    
        
    def add_param(self, name, reg, alpha):
        if not os.path.exists('models/' + name + '.pkl'):        
            self.tasks.append((name, self.data, reg, alpha))
        self.hyperparams.append((name, reg, alpha))

    def run(self, n_bootstrap=100):
        # print(self.tasks)
        # n_, d_, r_, a_ = self.tasks[0]
        # trainData(n_, d_, r_, 0.01)
        # print('done training')

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
        trainData(name, self.data, reg, alpha, test=True)

def random_risk_exp(n_cpus=None, n_bootstrap=30):
    m = Mimic2(mode='total', random_risk=True)
    ps = ParamSearch(m, n_cpus)

    reg = eye_loss    
    alphas = [0.1, 0.01, 0.001, 0.0001, 0.00001]

    for alpha in alphas:
        name = 'random_risk_eye' + '^' + str(alpha)
        ps.add_param(name, reg, alpha)

    ps.run(n_bootstrap)

def reg_exp(regs, n_cpus=None, n_bootstrap=30):
    m = Mimic2(mode='total')
    ps = ParamSearch(m, n_cpus)
    
    alphas = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    for reg in regs:
        for alpha in alphas:
            name = reg.__name__ + '^' + str(alpha)
            ps.add_param(name, reg, alpha)

    ps.run(n_bootstrap)

def duplicate_exp(regs, duplicate=3, n_cpus=None, n_bootstrap=30, noise=0.01):
    m = Mimic2(mode='total', duplicate=duplicate, noise=noise)
    ps = ParamSearch(m, n_cpus)
    
    alphas = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    for reg in regs:
        for alpha in alphas:
            name = reg.__name__ + '_dup' + str(duplicate) + '_' + str(noise)\
                   + '^' + str(alpha)
            ps.add_param(name, reg, alpha)

    ps.run(n_bootstrap)

def expert_feature_only_exp(n_cpus=None, n_bootstrap=30):    
    m = Mimic2(mode='total', expert_feature_only=True)
    ps = ParamSearch(m, n_cpus)

    reg = ridge
    alphas = [0.1, 0.01, 0.001, 0.0001, 0.00001]

    for alpha in alphas:
        name = 'expert_only_ridge' + '^' + str(alpha)
        ps.add_param(name, reg, alpha)

    ps.run(n_bootstrap)

def two_stage_exp(threshold=0.90, n_cpus=None, n_bootstrap=30):
    '''
    remove features by setting a threshold on correlation, 
    then apply l2 regularization on the remaining features
    '''
    m = Mimic2(mode='total', two_stage=True, threshold=float(threshold))
    ps = ParamSearch(m, n_cpus)

    reg = ridge
    alphas = [0.1, 0.01, 0.001, 0.0001, 0.00001]

    for alpha in alphas:
        name = 'two_stage_ridge_' + str(threshold) + '^' + str(alpha)
        ps.add_param(name, reg, alpha)

    ps.run(n_bootstrap)
    
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
        print(sys.argv[1:])
        f = eval(sys.argv[1])
        if len(sys.argv) >= 3:
            args = eval(sys.argv[2])
            f(args)
        else:
            f()
